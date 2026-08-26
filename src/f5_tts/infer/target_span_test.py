from __future__ import annotations

import unittest

import torch

from f5_tts.infer.target_span import (
    blend_vad_targets,
    crop_waveform,
    span_to_sample_bounds,
    target_latent_mask,
)


class TargetSpanTest(unittest.TestCase):
    def test_span_bounds_are_clamped_with_context(self):
        self.assertEqual(
            span_to_sample_bounds(
                (0.1, 0.2),
                sample_rate=10,
                num_samples=10,
                context_sec=0.5,
            ),
            (0, 7),
        )

    def test_crop_keeps_gradient_path(self):
        wav = torch.arange(20.0, requires_grad=True)
        crop = crop_waveform(wav, (0.4, 1.0), sample_rate=10)
        crop.sum().backward()
        self.assertEqual(tuple(crop.shape), (6,))
        self.assertTrue(torch.equal(
            wav.grad,
            torch.tensor([0.0] * 4 + [1.0] * 6 + [0.0] * 10),
        ))

    def test_blend_vad_targets(self):
        neutral = torch.tensor([[0.2, 0.3, 0.4]])
        emotion = torch.tensor([[0.6, 0.5, 0.8]])
        self.assertTrue(torch.allclose(
            blend_vad_targets(neutral, emotion, 0.5),
            torch.tensor([[0.4, 0.4, 0.6]]),
        ))

    def test_local_latent_mask_excludes_prompt_and_context(self):
        mask = target_latent_mask(
            span=(0.4, 0.8),
            batch_size=1,
            latent_length=20,
            prompt_length=4,
            sample_rate=10,
            latent_hop_length=1,
            context_sec=0.1,
            device="cpu",
        )
        self.assertEqual(mask.shape, (1, 20, 1))
        self.assertTrue(mask[0, 7:13].all())
        self.assertFalse(mask[0, :4].any())
        self.assertFalse(mask[0, 4:7].any())
        self.assertFalse(mask[0, 13:].any())


if __name__ == "__main__":
    unittest.main()
