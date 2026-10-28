"""CPU regressions for shifted-window boundaries and frame pairing.

Run with the model dependencies installed:
    python -m unittest discover -s tests -v
"""

import contextlib
import io
import unittest

import torch

from model.SW_HSS3D import SW_HSS3D
from model.csms6s import selective_scan_torch


class ShiftedWindowTests(unittest.TestCase):
    def make_model(self, *, window=8, shift=(4, 4), conv_first=True, zigzag=False,
                   shift_mode="padding", bias=False):
        torch.manual_seed(42)
        with contextlib.redirect_stdout(io.StringIO()):
            model = SW_HSS3D(
                d_model=4,
                d_state=2,
                expand=1,
                window_size=(window, window),
                shift_size=shift,
                conv_first=conv_first,
                zigzag=zigzag,
                shift_mode=shift_mode,
                bias=bias,
            )
        model.selective_scan = selective_scan_torch
        return model

    def test_mask_retains_every_real_feature(self):
        for window in (8, 16):
            for height, width in ((16, 24), (13, 17), (1, 5), (5, 1), (9, 9)):
                for shift in ((0, 0), (window // 2, window // 2), (0, window // 2), (window // 2, 0)):
                    with self.subTest(window=window, shape=(height, width), shift=shift):
                        model = self.make_model(window=window, shift=shift)
                        (pt, pb), (pl, pr) = model.window_pad(height, width)
                        mask = model.scan_mask(4, height, width, torch.device('cpu'))
                        if mask is None:
                            self.assertEqual((pt, pb, pl, pr), (0, 0, 0, 0))
                            continue
                        recovered = model.cross_merge(mask, height + pt + pb, width + pl + pr) / 4
                        real = recovered[:, pt:pt + height, pl:pl + width]
                        torch.testing.assert_close(real, torch.ones_like(real), atol=0, rtol=0)
                        self.assertEqual(recovered.sum().item(), 4 * height * width)
                        self.assertIs(mask, model.scan_mask(4, height, width, torch.device('cpu')))

    def test_corner_exchanges_frames_without_opposite_edge_or_other_samples(self):
        for conv_first in (False, True):
            for zigzag in (False, True):
                with self.subTest(conv_first=conv_first, zigzag=zigzag):
                    model = self.make_model(conv_first=conv_first, zigzag=zigzag)
                    # Batch layout: frame0/sample0, frame0/sample1,
                    #               frame1/sample0, frame1/sample1.
                    x = torch.randn(4, 16, 16, 4, requires_grad=True)
                    y = model(x)
                    y[0, :4, :4].square().sum().backward()
                    self.assertGreater(x.grad[2, 0, 0].abs().sum().item(), 0)
                    self.assertEqual(x.grad[[0, 2], 12:, 12:].count_nonzero().item(), 0)
                    self.assertEqual(x.grad[[1, 3]].count_nonzero().item(), 0)

    def test_forward_and_backward_on_odd_and_small_inputs(self):
        for conv_first in (False, True):
            for height, width in ((13, 17), (1, 5), (5, 1)):
                for shift in ((4, 4), (0, 4), (4, 0)):
                    with self.subTest(conv_first=conv_first, shape=(height, width), shift=shift):
                        model = self.make_model(conv_first=conv_first, shift=shift)
                        x = torch.randn(4, height, width, 4, requires_grad=True)
                        y = model(x, pad_token=torch.tensor([0.1, -0.2, 0.3, -0.4]))
                        self.assertEqual(y.shape, x.shape)
                        self.assertTrue(torch.isfinite(y).all().item())
                        y.square().mean().backward()
                        self.assertTrue(torch.isfinite(x.grad).all().item())
                        for parameter in model.parameters():
                            self.assertIsNotNone(parameter.grad)
                            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_roll_matches_padding_outputs_and_gradients(self):
        # Non-half shifts exercise Hilbert paths that leave and revisit a
        # boundary region; resetting on each region transition is insufficient.
        cases = (
            (2, 1, 1, (2, 2)),
            (2, 1, 7, (0, 2)),
            (4, 7, 1, (2, 0)),
            (4, 9, 9, (2, 2)),
            (4, 7, 11, (1, 3)),
            (2, 8, 12, (0, 0)),
            (2, 13, 17, (0, 2)),
        )
        for conv_first in (False, True):
            for zigzag in (False, True):
                for batch, height, width, shift in cases:
                    with self.subTest(conv_first=conv_first, zigzag=zigzag,
                                      shape=(batch, height, width), shift=shift):
                        options = dict(window=4, shift=shift, conv_first=conv_first,
                                       zigzag=zigzag, bias=True)
                        padding = self.make_model(**options)
                        roll = self.make_model(**options, shift_mode="roll")
                        roll.load_state_dict(padding.state_dict(), strict=True)
                        x = torch.randn(batch, height, width, 4, requires_grad=True)
                        xr = x.detach().clone().requires_grad_()
                        token = torch.randn(4, requires_grad=True)
                        token_r = token.detach().clone().requires_grad_()
                        y, yr = padding(x, token), roll(xr, token_r)
                        torch.testing.assert_close(yr, y, atol=2e-6, rtol=2e-5)
                        probe = torch.randn_like(y) / y.numel()
                        (y * probe).sum().backward()
                        (yr * probe).sum().backward()
                        torch.testing.assert_close(xr.grad, x.grad, atol=5e-7, rtol=2e-4)
                        if token.grad is None:
                            self.assertIsNone(token_r.grad)
                        else:
                            torch.testing.assert_close(token_r.grad, token.grad, atol=5e-7, rtol=2e-4)
                        for name, parameter in padding.named_parameters():
                            other = dict(roll.named_parameters())[name]
                            torch.testing.assert_close(other.grad, parameter.grad, atol=5e-7, rtol=2e-4)

    def test_roll_keeps_opposite_corners_and_samples_independent(self):
        for conv_first in (False, True):
            for zigzag in (False, True):
                with self.subTest(conv_first=conv_first, zigzag=zigzag):
                    model = self.make_model(conv_first=conv_first, zigzag=zigzag,
                                            shift_mode="roll")
                    x = torch.randn(4, 16, 16, 4, requires_grad=True)
                    model(x)[0, :4, :4].square().sum().backward()
                    self.assertGreater(x.grad[2, 0, 0].abs().sum().item(), 0)
                    self.assertEqual(x.grad[[0, 2], 12:, 12:].count_nonzero().item(), 0)
                    self.assertEqual(x.grad[[1, 3]].count_nonzero().item(), 0)

    def test_invalid_shift_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shift_mode"):
            self.make_model(shift_mode="unknown")

    def test_inference_cache_can_be_reused_for_training(self):
        for shift_mode in ("padding", "roll"):
            for conv_first in (False, True):
                with self.subTest(shift_mode=shift_mode, conv_first=conv_first):
                    model = self.make_model(shift_mode=shift_mode, conv_first=conv_first)
                    model.eval()
                    with torch.inference_mode():
                        model(torch.randn(2, 13, 17, 4))
                    model.train()
                    x = torch.randn(2, 13, 17, 4, requires_grad=True)
                    model(x).square().mean().backward()
                    self.assertTrue(torch.isfinite(x.grad).all().item())
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_roll_plan_cache_can_be_reused_with_different_batch_sizes(self):
        padding = self.make_model()
        roll = self.make_model(shift_mode="roll")
        roll.load_state_dict(padding.state_dict(), strict=True)
        plan = roll._roll_scan_plan(13, 17, torch.device('cpu'))
        for batch in (2, 6, 4):
            with self.subTest(batch=batch):
                x = torch.randn(batch, 13, 17, 4)
                torch.testing.assert_close(roll(x), padding(x), atol=2e-6, rtol=2e-5)
                self.assertIs(plan, roll._roll_scan_plan(13, 17, torch.device('cpu')))


if __name__ == '__main__':
    unittest.main()
