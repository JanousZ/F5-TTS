from __future__ import annotations

import unittest

from f5_tts.eval.align_local_emo_pilot import split_words, target_word_indices


class AlignLocalEmoPilotTest(unittest.TestCase):
    def test_split_words_keeps_char_offsets(self):
        words = split_words("Please send the final report after lunch.")
        self.assertEqual([w.normalized for w in words], [
            "please", "send", "the", "final", "report", "after", "lunch",
        ])
        self.assertEqual((words[3].start_char, words[3].end_char), (16, 21))

    def test_target_char_span_selects_word(self):
        row = {
            "gen_text": "Please send the final report after lunch.",
            "target_word": "final",
            "target_char_span": [16, 21],
        }
        self.assertEqual(target_word_indices(row, split_words(row["gen_text"])), [3])

    def test_repeated_word_requires_char_span(self):
        row = {"gen_text": "small small detail", "target_word": "small"}
        with self.assertRaises(ValueError):
            target_word_indices(row, split_words(row["gen_text"]))


if __name__ == "__main__":
    unittest.main()
