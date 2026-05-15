"""
forest_env.py  –  Scene-II Analogue (Cylindrical Forest / Pole Fields)
───────────────────────────────────────────────────────────────────────
Generates cylindrical-obstacle environments mimicking dense pole forests.
Features:
  • Poisson-disc sampling for natural spacing (no grid artefacts)
  • Spatially correlated radius variation (large trunks cluster together)
  • Height-banding: short shrubs + mid-canopy cylinders + tall masts
  • Cluster sub-patterns: rings, spirals, starburst, random Gaussian blobs
  • Narrow winding passages between cluster groups
  • Optional trunk colour gradient by height band (matches paper's rainbow vis.)

Difficulty scaling
──────────────────
  TRIVIAL  → ~30 cylinders, radii 0.4-0.8 m, wide paths
  EASY     → ~60 cylinders, mixed radii
  MEDIUM   → ~120 cylinders, tight clusters
  HARD     → ~200 cylinders, ringed sub-forests + spiral traps
  EXTREME  → ~300+ cylinders, near-impassable density
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Optional, Tuple

from .base_env import (
    BaseEnvironment, EnvironmentConfig, Obstacle,
    ObstacleType, DifficultyLevel, _PlacementGrid,
    sample_height, random_color,
)


_DIFFICULTY_PARAMS = {
    DifficultyLevel.TRIVIAL:  dict(n_target=30,  r_min=0.4, r_max=0.8,  n_clusters=2, cluster_mode="random",  ring_traps=0),
    DifficultyLevel.EASY:     dict(n_target=60,  r_min=0.3, r_max=1.0,  n_clusters=4, cluster_mode="blob",    ring_traps=0),
    DifficultyLevel.MEDIUM:   dict(n_target=120, r_min=0.3, r_max=1.2,  n_clusters=6, cluster_mode="mixed",   ring_traps=1),
    DifficultyLevel.HARD:     dict(n_target=200, r_min=0.25,r_max=1.5,  n_clusters=8, cluster_mode="ring",    ring_traps=2),
    DifficultyLevel.EXTREME:  dict(n_target=320, r_min=0.2, r_max=1.8,  n_clusters=12,cluster_mode="spiral",  ring_traps=3),
}


class CylindricalForestEnvironment(BaseEnvironment):
    """
    Cylindrical (pole/forest) obstacle environment.

    Parameters
    ──────────
    n_target     : int   – number of cylinders to attempt to place
    r_min/r_max  : float – cylinder radius range (m)
    n_clusters   : int   – number of spatial sub-clusters
    cluster_mode : str   – "random" | "blob" | "ring" | "spiral" | "mixed"
    ring_traps   : int   – number of closed cylinder rings (local minima)
    """

    def __init__(self, config: EnvironmentConfig,
                 n_target:    Optional[int]   = None,
                 r_min:       Optional[float] = None,
                 r_max:       Optional[float] = None,
                 n_clusters:  Optional[int]   = None,
                 cluster_mode: Optional[str]  = None,
                 ring_traps:  Optional[int]   = None):

        p = _DIFFICULTY_PARAMS[config.difficulty]
        self._n_target    = n_target    or p["n_target"]
        self._r_min       = r_min       or p["r_min"]
        self._r_max       = r_max       or p["r_max"]
        self._n_clusters  = n_clusters  or p["n_clusters"]
        self._cluster_mode = cluster_mode or p["cluster_mode"]
        self._ring_traps  = ring_traps  if ring_traps is not None else p["ring_traps"]
        super().__init__(config)

    # ──────────────────────────────────────────────────────────────────────────

    def _generate_obstacles(self):
        cfg = self.config
        rng = self.rng
        W, D = cfg.map_width, cfg.map_depth

        pgrid = _PlacementGrid(W, D, cell=0.5)
        placed = 0

        # ── 1. Generate cluster centres ───────────────────────────────────────
        cluster_centres = self._make_cluster_centres(rng, W, D)

        # ── 2. Fill clusters with cylinders ───────────────────────────────────
        attempts = 0
        max_attempts = self._n_target * 20

        while placed < self._n_target and attempts < max_attempts:
            attempts += 1
            # Pick a cluster centre and sample near it
            cc = cluster_centres[rng.integers(0, len(cluster_centres))]
            spread = rng.uniform(2.0, min(W, D) * 0.15)
            x = float(np.clip(rng.normal(cc[0], spread), self._r_max + 0.5, W - self._r_max - 0.5))
            y = float(np.clip(rng.normal(cc[1], spread), self._r_max + 0.5, D - self._r_max - 0.5))

            # Radius: spatially correlated – larger near cluster centre
            dist_to_cc = math.hypot(x - cc[0], y - cc[1])
            r_frac = max(0.0, 1.0 - dist_to_cc / (spread * 2.0))
            radius = float(self._r_min + r_frac * (self._r_max - self._r_min)
                           * rng.uniform(0.7, 1.3))
            radius = float(np.clip(radius, self._r_min, self._r_max))

            min_sep = radius + self._r_max + cfg.min_clearance
            if not pgrid.is_free(x, y, min_sep):
                continue

            h = float(sample_height(rng, cfg, 1)[0])
            # Height band → colour
            color = self._height_band_color(h, cfg.min_height, cfg.max_height, rng)

            pgrid.mark(x, y, radius + cfg.min_clearance * 0.5)
            self.obstacles.append(Obstacle(
                obstacle_type=ObstacleType.CYLINDER,
                position=np.array([x, y, 0.0]),
                dimensions=np.array([radius, radius, h]),
                color=color,
                metadata={
                    "cluster_id": int(rng.integers(0, self._n_clusters)),
                    "radius": radius,
                    "height_band": self._band_name(h, cfg.min_height, cfg.max_height),
                },
            ))
            placed += 1

        # ── 3. Ring traps (closed cylinder rings = local minima) ──────────────
        for _ in range(self._ring_traps):
            self._inject_ring_trap(pgrid, rng, cfg, W, D)

    # ── Cluster centre strategies ─────────────────────────────────────────────

    def _make_cluster_centres(self, rng, W, D):
        mode = self._cluster_mode
        n    = self._n_clusters
        centres = []

        if mode == "random" or mode not in ("blob", "ring", "spiral", "mixed"):
            for _ in range(n):
                centres.append((rng.uniform(W * 0.1, W * 0.9),
                                rng.uniform(D * 0.1, D * 0.9)))

        elif mode == "blob":
            # Gaussian blobs arranged on a macro grid
            macro_rows = max(1, int(math.sqrt(n)))
            macro_cols = math.ceil(n / macro_rows)
            for row in range(macro_rows):
                for col in range(macro_cols):
                    if len(centres) >= n:
                        break
                    cx = W * (col + 1) / (macro_cols + 1)
                    cy = D * (row + 1) / (macro_rows + 1)
                    centres.append((cx + rng.normal(0, W * 0.05),
                                    cy + rng.normal(0, D * 0.05)))

        elif mode == "ring":
            # Clusters arranged in a large ring
            for i in range(n):
                angle = 2 * math.pi * i / n
                cx = W / 2 + math.cos(angle) * W * 0.35
                cy = D / 2 + math.sin(angle) * D * 0.35
                centres.append((cx, cy))

        elif mode == "spiral":
            for i in range(n):
                t = 2 * math.pi * i / n * 2
                r = (W * 0.1) + t / (2 * math.pi) * (W * 0.35 / 2)
                cx = W / 2 + math.cos(t) * r
                cy = D / 2 + math.sin(t) * r
                centres.append((np.clip(cx, W*0.05, W*0.95),
                                np.clip(cy, D*0.05, D*0.95)))

        elif mode == "mixed":
            # Half blob, half ring
            blob_n = n // 2
            ring_n = n - blob_n
            for i in range(blob_n):
                centres.append((rng.uniform(W*0.1, W*0.9),
                                rng.uniform(D*0.1, D*0.9)))
            for i in range(ring_n):
                angle = 2 * math.pi * i / ring_n
                cx = W / 2 + math.cos(angle) * W * 0.3
                cy = D / 2 + math.sin(angle) * D * 0.3
                centres.append((cx, cy))

        return centres

    # ── Ring trap injection ───────────────────────────────────────────────────

    def _inject_ring_trap(self, pgrid, rng, cfg, W, D):
        """
        Place a ring of cylinders enclosing an area.
        Agents that wander in must detect the trap and find the gap.
        """
        cx = rng.uniform(W * 0.2, W * 0.8)
        cy = rng.uniform(D * 0.2, D * 0.8)
        ring_r    = rng.uniform(3.0, 6.0)
        n_posts   = rng.integers(8, 16)
        gap_idx   = int(rng.integers(0, n_posts))   # one gap for exit
        post_r    = rng.uniform(self._r_min, self._r_min * 2)
        h         = cfg.max_height * 0.9

        for i in range(n_posts):
            if i == gap_idx:
                continue   # leave one opening
            angle = 2 * math.pi * i / n_posts
            px = cx + math.cos(angle) * ring_r
            py = cy + math.sin(angle) * ring_r
            px = float(np.clip(px, post_r + 0.5, W - post_r - 0.5))
            py = float(np.clip(py, post_r + 0.5, D - post_r - 0.5))
            pgrid.mark(px, py, post_r + 0.3)
            self.obstacles.append(Obstacle(
                obstacle_type=ObstacleType.CYLINDER,
                position=np.array([px, py, 0.0]),
                dimensions=np.array([post_r, post_r, h]),
                color=(1.0, 0.2, 0.2),
                metadata={"layer": "ring_trap", "ring_cx": cx, "ring_cy": cy},
            ))

    # ── Colour helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _height_band_color(h, h_min, h_max, rng):
        """Map height to a colour band (shrub→green, mid→cyan, tall→magenta/red)."""
        span = max(h_max - h_min, 1e-6)
        t    = (h - h_min) / span
        if t < 0.33:
            return (rng.uniform(0.1, 0.3), rng.uniform(0.6, 0.9), rng.uniform(0.1, 0.4))  # green
        elif t < 0.66:
            return (rng.uniform(0.1, 0.4), rng.uniform(0.6, 0.9), rng.uniform(0.7, 1.0))  # cyan
        else:
            return (rng.uniform(0.6, 1.0), rng.uniform(0.1, 0.4), rng.uniform(0.6, 1.0))  # magenta

    @staticmethod
    def _band_name(h, h_min, h_max):
        t = (h - h_min) / max(h_max - h_min, 1e-6)
        return "shrub" if t < 0.33 else ("canopy" if t < 0.66 else "mast")
