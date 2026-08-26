import unittest

import torch

from f5_tts.infer.emotion_vad_target import emotion_vad_target


class EmotionVADTargetTest(unittest.TestCase):
    def test_anchor_order_is_a_d_v(self):
        self.assertTrue(torch.allclose(
            emotion_vad_target("happy"),
            torch.tensor([0.735, 0.772, 1.0]),
            atol=1e-6,
        ))

    def test_alpha_half_interpolates_from_neutral(self):
        self.assertTrue(torch.allclose(
            emotion_vad_target("sad", alpha=0.5),
            torch.tensor([0.4165, 0.3245, 0.3625]),
            atol=1e-6,
        ))

    def test_unknown_emotion_fails(self):
        with self.assertRaises(ValueError):
            emotion_vad_target("neutral")

if __name__ == "__main__":
    unittest.main()
