"""
urban_env.py  –  Scene-I Analogue (and beyond)
────────────────────────────────────────────────
Generates dense urban-block environments with:
  • Varying building heights (uniform / normal / bimodal distributions)
  • Mandatory narrow corridors and dead-end streets
  • Bottleneck intersections
  • Configurable block-grid and random perturbations
  • Multi-storey "super-tall" landmark buildings
  • Optional inner-city plaza (trap / local-minima region)

Difficulty scaling
──────────────────
  TRIVIAL  → sparse grid, wide corridors (≥ 5 m)
  EASY     → baseline paper comparable
  MEDIUM   → tighter grid, sub-3-m corridors emerge
  HARD     → mandatory dead-ends + bottleneck gates
  EXTREME  → near-impassable dense grid + random partial-fills
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


# ─────────────────────────────────────────────────────────────────────────────
# Tuning tables indexed by DifficultyLevel
# ─────────────────────────────────────────────────────────────────────────────

_DIFFICULTY_PARAMS = {
    DifficultyLevel.TRIVIAL: dict(
        grid_rows=3, grid_cols=3, jitter=0.2,
        corridor_w=6.0, density=0.10,
        n_dead_ends=0, n_bottlenecks=0,
        supertall_frac=0.0, perturb_extra=0),

    DifficultyLevel.EASY: dict(
        grid_rows=4, grid_cols=4, jitter=0.3,
        corridor_w=4.5, density=0.20,
        n_dead_ends=1, n_bottlenecks=1,
        supertall_frac=0.05, perturb_extra=3),

    DifficultyLevel.MEDIUM: dict(
        grid_rows=6, grid_cols=6, jitter=0.35,
        corridor_w=3.5, density=0.30,
        n_dead_ends=3, n_bottlenecks=2,
        supertall_frac=0.10, perturb_extra=8),

    DifficultyLevel.HARD: dict(
        grid_rows=8, grid_cols=8, jitter=0.40,
        corridor_w=2.5, density=0.40,
        n_dead_ends=6, n_bottlenecks=4,
        supertall_frac=0.15, perturb_extra=15),

    DifficultyLevel.EXTREME: dict(
        grid_rows=10, grid_cols=10, jitter=0.45,
        corridor_w=1.8, density=0.55,
        n_dead_ends=10, n_bottlenecks=7,
        supertall_frac=0.20, perturb_extra=25),
}


class DenseUrbanEnvironment(BaseEnvironment):
    """
    Dense urban-block environment.

    Key parameters
    ──────────────
    grid_rows / grid_cols : int
        Base grid resolution. Buildings are placed at grid intersections then
        jittered for visual realism.
    corridor_w : float
        Target corridor width (metres). Narrower = harder.
    n_dead_ends : int
        Number of L-shaped or U-shaped dead-end regions to inject.
    n_bottlenecks : int
        Number of narrow gateway obstacles added across major corridors.
    supertall_frac : float
        Fraction of buildings assigned 2-3× the nominal max height.
    perturb_extra : int
        Additional randomly-placed buildings beyond the grid.
    """

    def __init__(self, config: EnvironmentConfig,
                 grid_rows: Optional[int] = None,
                 grid_cols: Optional[int] = None,
                 corridor_w: Optional[float] = None,
                 n_dead_ends: Optional[int] = None,
                 n_bottlenecks: Optional[int] = None,
                 supertall_frac: Optional[float] = None,
                 perturb_extra: Optional[int] = None):

        params = _DIFFICULTY_PARAMS[config.difficulty].copy()
        self._grid_rows     = grid_rows    or params["grid_rows"]
        self._grid_cols     = grid_cols    or params["grid_cols"]
        self._jitter        = params["jitter"]
        self._corridor_w    = corridor_w   or params["corridor_w"]
        self._n_dead_ends   = n_dead_ends  if n_dead_ends  is not None else params["n_dead_ends"]
        self._n_bottlenecks = n_bottlenecks if n_bottlenecks is not None else params["n_bottlenecks"]
        self._supertall_frac = supertall_frac if supertall_frac is not None else params["supertall_frac"]
        self._perturb_extra = perturb_extra if perturb_extra is not None else params["perturb_extra"]

        super().__init__(config)

    # ──────────────────────────────────────────────────────────────────────────

    def _generate_obstacles(self):
        cfg  = self.config
        rng  = self.rng
        W, D = cfg.map_width, cfg.map_depth

        # ── 1. Grid-based building placement ─────────────────────────────────
        # Cell dimensions (building centre spacing)
        cell_w = W / (self._grid_cols + 1)
        cell_d = D / (self._grid_rows + 1)

        # Building footprint = cell minus corridor
        b_w = max(0.5, cell_w - self._corridor_w)
        b_d = max(0.5, cell_d - self._corridor_w)

        pgrid = _PlacementGrid(W, D, cell=0.5)
        heights = sample_height(rng, cfg, self._grid_rows * self._grid_cols)
        idx = 0

        # Supertall mask
        n_buildings = self._grid_rows * self._grid_cols
        supertall_mask = rng.random(n_buildings) < self._supertall_frac

        for row in range(self._grid_rows):
            for col in range(self._grid_cols):
                # Nominal centre
                cx = cell_w * (col + 1) + rng.uniform(-cell_w * self._jitter,
                                                        cell_w * self._jitter)
                cy = cell_d * (row + 1) + rng.uniform(-cell_d * self._jitter,
                                                        cell_d * self._jitter)
                cx = np.clip(cx, b_w / 2 + 0.5, W - b_w / 2 - 0.5)
                cy = np.clip(cy, b_d / 2 + 0.5, D - b_d / 2 - 0.5)

                h = heights[idx]
                if supertall_mask[idx]:
                    h = min(h * rng.uniform(2.0, 3.0), cfg.map_height * 0.95)

                # Footprint jitter
                fw = b_w * rng.uniform(0.6, 1.2)
                fd = b_d * rng.uniform(0.6, 1.2)

                r = max(fw, fd) / 2.0 + cfg.min_clearance
                if pgrid.is_free(cx, cy, r * 0.8):
                    pgrid.mark(cx, cy, r * 0.8)
                    self.obstacles.append(Obstacle(
                        obstacle_type=ObstacleType.CUBOID,
                        position=np.array([cx, cy, 0.0]),
                        dimensions=np.array([fw, fd, h]),
                        rotation_deg=rng.uniform(-15, 15) if not supertall_mask[idx] else 0.0,
                        color=self._building_color(rng, supertall_mask[idx]),
                        metadata={"layer": "grid", "supertall": bool(supertall_mask[idx]),
                                  "row": row, "col": col},
                    ))
                idx += 1

        # ── 2. Extra perturb buildings ────────────────────────────────────────
        extra_heights = sample_height(rng, cfg, self._perturb_extra)
        for i in range(self._perturb_extra):
            cx = rng.uniform(2.0, W - 2.0)
            cy = rng.uniform(2.0, D - 2.0)
            fw = rng.uniform(1.0, b_w)
            fd = rng.uniform(1.0, b_d)
            r  = max(fw, fd) / 2.0 + cfg.min_clearance * 0.5
            if pgrid.is_free(cx, cy, r * 0.7):
                pgrid.mark(cx, cy, r * 0.7)
                self.obstacles.append(Obstacle(
                    obstacle_type=ObstacleType.CUBOID,
                    position=np.array([cx, cy, 0.0]),
                    dimensions=np.array([fw, fd, float(extra_heights[i])]),
                    rotation_deg=rng.uniform(-30, 30),
                    color=self._building_color(rng, False),
                    metadata={"layer": "perturb"},
                ))

        # ── 3. Inject dead-end regions ────────────────────────────────────────
        self._inject_dead_ends(pgrid, rng, cfg, cell_w, cell_d)

        # ── 4. Inject bottleneck gates ────────────────────────────────────────
        self._inject_bottlenecks(pgrid, rng, cfg)

    # ── Dead-end injection ────────────────────────────────────────────────────

    def _inject_dead_ends(self, pgrid, rng, cfg, cell_w, cell_d):
        """
        Place U-shaped or L-shaped clusters that form dead-end pockets.
        Agents must detect the dead end and reverse rather than proceeding.
        """
        W, D = cfg.map_width, cfg.map_depth
        wall_h = cfg.max_height * 0.8

        for _ in range(self._n_dead_ends):
            # Pick a free centre region for the dead end
            cx = rng.uniform(cell_w * 2, W - cell_w * 2)
            cy = rng.uniform(cell_d * 2, D - cell_d * 2)
            opening_dir = rng.choice(["N", "S", "E", "W"])
            depth  = rng.uniform(3.0, 6.0)
            width  = rng.uniform(2.5, 5.0)
            thick  = rng.uniform(0.8, 1.5)

            walls = self._u_shape_walls(cx, cy, opening_dir, depth, width, thick)
            for (wx, wy, ww, wd) in walls:
                wx = np.clip(wx, thick, W - thick)
                wy = np.clip(wy, thick, D - thick)
                r  = max(ww, wd) / 2.0
                if pgrid.is_free(wx, wy, r * 0.6):
                    pgrid.mark(wx, wy, r * 0.6)
                    self.obstacles.append(Obstacle(
                        obstacle_type=ObstacleType.CUBOID,
                        position=np.array([wx, wy, 0.0]),
                        dimensions=np.array([ww, wd, wall_h]),
                        color=(0.3, 0.3, 0.3),
                        metadata={"layer": "dead_end"},
                    ))

    @staticmethod
    def _u_shape_walls(cx, cy, direction, depth, width, thick):
        """Return list of (x, y, w, d) tuples forming a U-shape."""
        half_w = width / 2.0
        # Back wall
        if direction in ("N", "S"):
            sign = 1.0 if direction == "N" else -1.0
            back = (cx, cy + sign * depth / 2.0, width, thick)
            left = (cx - half_w + thick / 2.0, cy, thick, depth)
            right = (cx + half_w - thick / 2.0, cy, thick, depth)
        else:
            sign = 1.0 if direction == "E" else -1.0
            back = (cx + sign * depth / 2.0, cy, thick, width)
            left = (cx, cy - half_w + thick / 2.0, depth, thick)
            right = (cx, cy + half_w - thick / 2.0, depth, thick)
        return [back, left, right]

    # ── Bottleneck injection ──────────────────────────────────────────────────

    def _inject_bottlenecks(self, pgrid, rng, cfg):
        """
        Place pairs of tall wall segments that narrow a corridor to ~1 UAV width.
        Forces agents to queue, inducing coordination conflicts.
        """
        W, D = cfg.map_width, cfg.map_depth
        for _ in range(self._n_bottlenecks):
            # Horizontal or vertical gate
            horizontal = rng.random() > 0.5
            if horizontal:
                gate_x = rng.uniform(W * 0.2, W * 0.8)
                gate_y = rng.uniform(D * 0.3, D * 0.7)
                gap    = rng.uniform(1.5, 2.5)           # width of opening
                arm_l  = rng.uniform(2.0, 5.0)           # length of each wall arm
                h      = cfg.max_height
                # Two wall arms flanking the gap
                for sign in [-1, 1]:
                    arm_y = gate_y + sign * (gap / 2.0 + arm_l / 2.0)
                    if pgrid.is_free(gate_x, arm_y, 0.8):
                        pgrid.mark(gate_x, arm_y, 0.8)
                        self.obstacles.append(Obstacle(
                            obstacle_type=ObstacleType.CUBOID,
                            position=np.array([gate_x, arm_y, 0.0]),
                            dimensions=np.array([1.0, arm_l, h]),
                            color=(0.2, 0.2, 0.6),
                            metadata={"layer": "bottleneck"},
                        ))
            else:
                gate_y = rng.uniform(D * 0.2, D * 0.8)
                gate_x = rng.uniform(W * 0.3, W * 0.7)
                gap    = rng.uniform(1.5, 2.5)
                arm_l  = rng.uniform(2.0, 5.0)
                h      = cfg.max_height
                for sign in [-1, 1]:
                    arm_x = gate_x + sign * (gap / 2.0 + arm_l / 2.0)
                    if pgrid.is_free(arm_x, gate_y, 0.8):
                        pgrid.mark(arm_x, gate_y, 0.8)
                        self.obstacles.append(Obstacle(
                            obstacle_type=ObstacleType.CUBOID,
                            position=np.array([arm_x, gate_y, 0.0]),
                            dimensions=np.array([arm_l, 1.0, h]),
                            color=(0.2, 0.2, 0.6),
                            metadata={"layer": "bottleneck"},
                        ))

    # ── Colour helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _building_color(rng, supertall: bool):
        if supertall:
            # Glass / steel – cyan-ish
            return (rng.uniform(0.5, 0.8), rng.uniform(0.7, 0.9), rng.uniform(0.8, 1.0))
        # Random warm / cool hue
        r = rng.uniform(0.3, 0.9)
        g = rng.uniform(0.3, 0.9)
        b = rng.uniform(0.3, 0.9)
        return (float(r), float(g), float(b))
