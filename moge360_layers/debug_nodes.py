"""
Debug/diagnostic nodes for MoGe360.
Provides quick visibility into depth map health and distribution.
"""

import torch
import numpy as np
import logging

import comfy.model_management as mm

log = logging.getLogger(__name__)


class Depth_Map_Diagnostics:
    """Analyze an ERP depth map and report useful statistics with a histogram preview."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "erp_depth": ("IMAGE", {
                    "tooltip": "ERP depth image [B,H,W,3]. Uses channel 0 as depth. Values should be non-negative with 0 meaning invalid, or a normalized visualization."
                }),
                "title": ("STRING", {
                    "default": "Depth Diagnostics",
                    "tooltip": "Optional label included in the summary for identification."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("summary", "hist_preview")
    OUTPUT_TOOLTIPS = (
        "Text summary of depth statistics: valid ratio, min/median/max, and percentiles.",
        "Histogram preview image visualizing the depth distribution and validity."
    )
    FUNCTION = "analyze_depth"
    CATEGORY = "MoGe360/Debug"
    DESCRIPTION = "Compute statistics and a small histogram preview for an ERP depth map to diagnose sky-only or collapsed-depth issues."

    def analyze_depth(self, erp_depth: torch.Tensor, title: str):
        device = mm.get_torch_device()

        try:
            # Expect [B,H,W,3]; analyze channel 0 of first batch
            d = erp_depth[0, :, :, 0].detach().cpu().float().numpy()
        except Exception as e:
            return (f"Invalid depth tensor shape {getattr(erp_depth, 'shape', None)}: {e}", self._blank_hist(device))

        H, W = d.shape

        # Basic masks and stats
        valid_mask = d > 0
        valid_ratio = float(valid_mask.mean()) if valid_mask.size > 0 else 0.0

        if valid_mask.any():
            vals = d[valid_mask]
            dmin = float(vals.min())
            dmax = float(vals.max())
            dmed = float(np.median(vals))
            pcts = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99]).astype(float)
        else:
            dmin = dmax = dmed = 0.0
            pcts = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)

        # Compact histogram image (grayscale, 256 bins)
        hist_img = self._make_hist_image(d, valid_mask)
        hist_tensor = torch.from_numpy(hist_img).unsqueeze(0).to(device)

        # Build summary string
        pct_labels = ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]
        pct_str = ", ".join([f"{n}:{v:.6g}" for n, v in zip(pct_labels, pcts)])

        summary_lines = [
            f"=== {title} ===",
            f"shape: {H}x{W}",
            f"valid_ratio: {valid_ratio:.3%}",
            f"min/median/max: {dmin:.6g} / {dmed:.6g} / {dmax:.6g}",
            f"percentiles: {pct_str}",
        ]

        return ("\n".join(summary_lines), hist_tensor)

    def _blank_hist(self, device):
        img = np.zeros((160, 256, 3), dtype=np.float32)
        return torch.from_numpy(img).unsqueeze(0).to(device)

    def _make_hist_image(self, depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        h, w = 160, 256
        img = np.zeros((h, w, 3), dtype=np.float32)

        if not valid_mask.any():
            # Draw a red bar to signal empty
            img[:, :, 0] = 0.3
            return img

        vals = depth[valid_mask]

        # Robust range for histogram
        p1, p99 = np.percentile(vals, [1, 99])
        if p99 <= p1:
            p1 = float(vals.min())
            p99 = float(vals.max())
            if p99 <= p1:
                p99 = p1 + 1.0

        # Compute histogram (256 bins)
        bins = np.linspace(p1, p99, w + 1)
        hist, _ = np.histogram(np.clip(vals, p1, p99), bins=bins)
        hist = hist.astype(np.float32)
        hist /= (hist.max() + 1e-6)

        # Draw bars (blue channel)
        for x in range(w):
            bar_h = int(hist[x] * (h - 1))
            if bar_h > 0:
                img[h - bar_h : h, x, 2] = 0.9

        # Overlay valid ratio (green stripe height)
        vr = float(valid_mask.mean())
        stripe_h = int(vr * h)
        if stripe_h > 0:
            img[0:stripe_h, 0:6, 1] = 0.6

        return img

