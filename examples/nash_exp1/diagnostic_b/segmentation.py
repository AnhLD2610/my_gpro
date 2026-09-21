# Copyright 2026 The nash_exp contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Blank-line partitions of original generated token positions."""

import codecs
from bisect import bisect_right
from collections.abc import Sequence


def segment_token_spans(text: str, token_spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return half-open token ranges; delimiters and crossing tokens stay left.

    Spans are character offsets in the decoded original sequence. They may
    overlap for tokens contributing bytes to the same Unicode character, and
    zero-width special tokens are retained. Leading empty paragraphs merge
    forward. No token is decoded or retokenized by this function.
    """
    if not token_spans:
        return []
    previous_start = 0
    for start, end in token_spans:
        if not (previous_start <= start <= end <= len(text)):
            raise ValueError("token spans must be ordered valid character offsets")
        previous_start = start
    cuts = []
    cursor = paragraph_start = 0
    while True:
        delimiter = text.find("\n\n", cursor)
        if delimiter < 0:
            break
        boundary = delimiter + 2
        has_content = bool(text[paragraph_start:delimiter].strip())
        if has_content and boundary < len(text):
            cuts.append(boundary)
            paragraph_start = boundary
        elif not has_content and cuts and paragraph_start == cuts[-1]:
            # Further delimiters after an empty paragraph still belong to the
            # preceding nonempty segment, including a trailing delimiter run.
            if boundary < len(text):
                cuts[-1] = boundary
                paragraph_start = boundary
            else:
                cuts.pop()
        cursor = boundary
    owners = [bisect_right(cuts, start) for start, _ in token_spans]
    result = []
    start = 0
    for index in range(1, len(owners)):
        if owners[index] != owners[index - 1]:
            result.append((start, index))
            start = index
    result.append((start, len(owners)))
    return result


def decode_original_token_spans(tokenizer, token_ids: Sequence[int]) -> tuple[str, list[tuple[int, int]]]:
    """Exact linear-time spans for byte-level tokenizers such as Qwen's.

    This decodes the original token spellings, including EOS/special tokens.
    It never calls encode or retokenizes text. A tokenizer whose decoder does
    not match byte-level decoding fails explicitly instead of inventing spans.
    Split UTF-8 characters may have overlapping token spans, as required to
    keep a boundary-crossing token entirely in its preceding segment.
    """
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    characters = byte_values.copy()
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            characters.append(256 + extra)
            extra += 1
    inverse = {chr(character): byte for byte, character in zip(byte_values, characters, strict=True)}
    token_ids = list(token_ids)
    spellings = tokenizer.convert_ids_to_tokens(token_ids)
    special_ids = set(tokenizer.all_special_ids)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    chunks = []
    spans = []
    character_count = 0
    for token_id, spelling in zip(token_ids, spellings, strict=True):
        if token_id in special_ids:
            chunk = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            payload = chunk.encode("utf-8")
        else:
            try:
                payload = bytes(inverse[character] for character in spelling)
            except (KeyError, TypeError) as exc:
                raise ValueError("UNSUPPORTED_TOKENIZER_DECODER: expected byte-level token spellings") from exc
        start = character_count
        decoded = decoder.decode(payload, final=False)
        chunks.append(decoded)
        character_count += len(decoded)
        pending_bytes = decoder.getstate()[0]
        spans.append((start, character_count + bool(pending_bytes)))
    chunks.append(decoder.decode(b"", final=True))
    text = "".join(chunks)
    expected = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if text != expected:
        raise ValueError("UNSUPPORTED_TOKENIZER_DECODER: byte-level reconstruction differs from original IDs")
    return text, [(min(start, len(text)), min(end, len(text))) for start, end in spans]
