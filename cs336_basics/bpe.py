import os
import regex as re
from typing import BinaryIO, List, Tuple
from collections import Counter, defaultdict
from multiprocessing import Pool, get_context

GPT2_PRETOKENIZE_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(task_args: List):
    input_file_path, split_special_tokens, start, end = task_args
    with open(input_file_path, "rb") as fid:
        fid.seek(start)
        text = fid.read(end - start).decode("utf-8")

    split_special_tokens.sort(key=len, reverse=True)
    match_special_token_pattern = re.compile(
        "|".join(map(re.escape, split_special_tokens))
    )
    match_gpt2_tokenizer_pattern = re.compile(GPT2_PRETOKENIZE_PATTERN)
    split_by_special_token_iter = match_special_token_pattern.split(text)
    pretokens2count = defaultdict(int)
    for document in split_by_special_token_iter:
        if document in split_special_tokens:
            continue
        pretokenized_document_iter = match_gpt2_tokenizer_pattern.finditer(document)
        for pretokenized_document in pretokenized_document_iter:
            pre_tokens = tuple(pretokenized_document.group(0).encode("utf-8"))
            pretokens2count[pre_tokens] += 1
    return pretokens2count


def train_bpe(input_file_path: str, vocab_size: int, special_tokens: List[str]):

    # Initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # pretokenization step
    # compute chunk boundaries
    num_processes = 4
    with open(input_file_path, "rb") as fid:
        boundaries = find_chunk_boundaries(fid, num_processes, b"<|endoftext|>")

    # Pretokenize
    task_args = [
        [input_file_path, special_tokens, start, end]
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with get_context("spawn").Pool(processes=num_processes) as pool:
        pretokens2count_list = pool.map(pretokenize_chunk, task_args)
    # pretokens2count_list = [pretokenize_chunk(task_args[0])]

    pretokens2count = Counter()
    for chunk_dict in pretokens2count_list:
        pretokens2count.update(chunk_dict)

    merges: List[Tuple[bytes, bytes]] = []
    # pair2pretokens_indices = defaultdict(lambda: defaultdict(list))
    pair2counts = defaultdict(int)
    pair2pretokens = defaultdict(set)
    for pre_token, count in pretokens2count.items():
        for i in range(len(pre_token) - 1):
            pair = pre_token[i : i + 2]
            pair2counts[pair] = pair2counts.get(pair, 0) + count
            pair2pretokens[pair].add(pre_token)
            # pair2pretokens_indices[pair][pre_token].append(i)

    count2pair = defaultdict(list)
    for pair, count in pair2counts.items():
        count2pair[count].append(pair)

    while len(vocab) < vocab_size and pair2counts:
        # find the top count pair
        top_count = 0
        for count in count2pair.keys():
            if top_count < count:
                top_count = count

        top_count_pairs = count2pair[top_count]
        top_count_pairs.sort(reverse=True)
        merge_pair = top_count_pairs.pop(0)
        if not top_count_pairs:
            count2pair.pop(top_count)
        # update the merge list
        merges.append(merge_pair)

        # Update vocab
        vocab[len(vocab)] = merge_pair
        affected_pretokens = pair2pretokens.pop(merge_pair)
        # pair2counts[merge_pair] = 0  # Should I remove this?
        for affected_pretoken in affected_pretokens:
            pretoken_count = pretokens2count[affected_pretoken]
            for i in range(len(affected_pretoken) - 1):
                pair = affected_pretoken[i : i + 2]

                # Step A - Discard old pairs
                pair2counts[pair] -= pretoken_count
                if pair in pair2pretokens:
                    pair2pretokens[pair].discard(affected_pretoken)
                    if not pair2pretokens[pair]:
                        pair2pretokens.pop(pair)

            # step B - Merge and rebuild the word
            i = 0
            new_word_list = []
            while i < len(affected_pretoken):
                # Check if we found the pair to merge
                if (
                    i < len(affected_pretoken) - 1
                    and affected_pretoken[i : i + 2] == merge_pair
                ):
                    # Append the merged token (bytes)
                    new_word_list.append(merge_pair)
                    i += 2  # Skip both parts
                else:
                    new_word_list.append(affected_pretoken[i])
                    i += 1
            new_word = tuple(new_word_list)

            # --- STEP C: INCREMENT NEW PAIRS ---
            # Now add the New Tuple to the index for all its pairs
            for i in range(len(new_word) - 1):
                pair = new_word[i : i + 2]
                pair2counts[pair] += pretoken_count
                pair2pretokens[pair].add(new_word)

            # --- STEP D: UPDATE MAIN COUNT ---
            # Swap the key in the main dictionary
            pretokens2count.pop(affected_pretoken)
            pretokens2count[
                new_word
            ] += pretoken_count  # Use += in case the merge created a word that already exists

    print(merges)
    return merges
