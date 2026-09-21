# Copyright 2026 The nash_exp contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
from diagnostic_b.segmentation import decode_original_token_spans, segment_token_spans


def char_spans(text):
    return [(i, i + 1) for i in range(len(text))]


def assert_partition(parts, token_count):
    assert [i for start, end in parts for i in range(start, end)] == list(range(token_count))
    assert all(end > start for start, end in parts)


@pytest.mark.parametrize("text", ["first\n\nsecond", "\n\nfirst\n\nsecond", "a\n\n\n\nb", "a\n\nb\n\n", "abc"])
def test_partition_every_token(text):
    parts = segment_token_spans(text, char_spans(text))
    assert_partition(parts, len(text))


def test_delimiter_belongs_to_preceding_segment():
    text = "a\n\nb"
    assert segment_token_spans(text, char_spans(text)) == [(0, 3), (3, 4)]


def test_consecutive_empty_paragraph_delimiters_stay_with_preceding_nonempty_segment():
    text = "a\n\n\n\nb"
    assert segment_token_spans(text, char_spans(text)) == [(0, 5), (5, 6)]
    text = "a\n\n\n\n"
    assert segment_token_spans(text, char_spans(text)) == [(0, len(text))]


def test_initial_empty_merges_forward():
    text = "\n\na\n\nb"
    assert segment_token_spans(text, char_spans(text)) == [(0, 5), (5, 6)]


def test_crossing_token_wholly_belongs_to_preceding_segment():
    text = "a\n\nbc"
    # The middle token includes the delimiter and the first character of paragraph 2.
    spans = [(0, 1), (1, 4), (4, 5)]
    assert segment_token_spans(text, spans) == [(0, 2), (2, 3)]


def test_single_token_crossing_multiple_boundaries():
    text = "a\n\nb\n\nc"
    spans = [(0, len(text))]
    assert segment_token_spans(text, spans) == [(0, 1)]


def test_special_tokens_zero_width_are_retained():
    text = "a\n\nb"
    spans = [(0, 0), (0, 1), (1, 3), (3, 4), (4, 4)]
    parts = segment_token_spans(text, spans)
    assert parts == [(0, 3), (3, 5)]
    assert_partition(parts, 5)


def test_invalid_offsets_rejected():
    with pytest.raises(ValueError):
        segment_token_spans("abc", [(2, 3), (1, 2)])


class ByteTokenizerFixture:
    all_special_ids = [9]
    spellings = {1: "a", 2: "ĊĊb", 3: "Ã", 4: "©", 9: "<eos>"}
    payloads = {1: b"a", 2: b"\n\nb", 3: b"\xc3", 4: b"\xa9", 9: b"<eos>"}

    def convert_ids_to_tokens(self, ids):
        return [self.spellings[index] for index in ids]

    def decode(self, ids, **kwargs):
        return b"".join(self.payloads[index] for index in ids).decode("utf-8", errors="replace")

    def encode(self, *args, **kwargs):
        raise AssertionError("Retokenizing generated IDs is forbidden")


def test_byte_decoder_original_ids_unicode_split_specials_and_crossing_boundary():
    text, spans = decode_original_token_spans(ByteTokenizerFixture(), [1, 2, 3, 4, 9])
    assert text == "a\n\nbé<eos>"
    assert spans == [(0, 1), (1, 4), (4, 5), (4, 5), (5, 10)]
    assert segment_token_spans(text, spans) == [(0, 2), (2, 5)]
