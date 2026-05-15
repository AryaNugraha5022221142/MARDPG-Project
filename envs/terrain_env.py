"""
terrain_env.py  –  Irregular Terrain Analogue
─────────────────────────────────────────────
Simulates unstructured/irregular terrain via overlapping spheres & warped cuboids.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Optional

from .base_env import (
    BaseEnvironment, EnvironmentConfig, Obstacle,
    ObstacleType, DifficultyLevel, _PlacementGrid,
    sample_height
)


_DIFFICULTY_PARAMS = {
    DifficultyLevel.TRIVIAL: dict(n_boulders=20, n_hills=2, max_r=5.0),
    DifficultyLevel.EASY:    dict(n_boulders=40, n_hills=4, max_r=6.0),
    DifficultyLevel.MEDIUM:  dict(n_boulders=80, n_hills=6, max_r=7.0),
    DifficultyLevel.HARD:    dict(n_boulders=140, n_hills=8,max_r=8.0),
    DifficultyLevel.EXTREME: dict(n_boulders=220, n_hills=10,max_r=10.0),
}


class IrregularTerrainEnvironment(BaseEnvironment):
    """
    Irregular Terrain Environment composed mostly of spherical and irregularly rotated cuboid obstacles to represent boulders, hills, and uneven ground.
    """

    def __init__(self, config: EnvironmentConfig,
                 n_boulders: Optional[int] = None,
                 n_hills: Optional[int] = None,
                 max_r: Optional[float] = None):

        p = _DIFFICULTY_PARAMS[config.difficulty]
        self._n_boulders = n_boulders or p["n_boulders"]
        self._n_hills    = n_hills    or p["n_hills"]
        self._max_r      = max_r      or p["max_r"]
        super().__init__(config)

    def _generate_obstacles(self):
        cfg = self.config
        rng = self.rng
        W, D = cfg.map_width, cfg.map_depth

        pgrid = _PlacementGrid(W, D, cell=1.0)
        
        # Hills (large overlapping spheres/cuboids at base)
        for _ in range(self._n_hills):
            cx = rng.uniform(W * 0.1, W * 0.9)
            cy = rng.uniform(D * 0.1, D * 0.9)
            r  = rng.uniform(self._max_r * 0.5, self._max_r)
            h  = rng.uniform(cfg.min_height, cfg.max_height * 0.8)
            
            pgrid.mark(cx, cy, r * 0.5)
            self.obstacles.append(Obstacle(
                obstacle_type=ObstacleType.SPHERE,
                position=np.array([cx, cy, 0.0]),
                dimensions=np.array([r*2, r*2, h]), # Treated as ellipsoid/sphere dimensions
                color=(0.5, 0.4, 0.3),
                metadata={"layer": "hill", "allow_overlap": True}
            ))

        # Boulders (scattered irregular obstacles)
        attempts = 0
        placed = 0
        while placed < self._n_boulders and attempts < self._n_boulders * 5:
            attempts += 1
            cx = rng.uniform(1.0, W - 1.0)
            cy = rng.uniform(1.0, D - 1.0)
            r = rng.uniform(0.5, self._max_r * 0.4)
            
            if not pgrid.is_free(cx, cy, r + cfg.min_clearance):
                continue
                
            pgrid.mark(cx, cy, r)
            
            # 50% sphere, 50% randomly rotated box
            if rng.random() > 0.5:
                # Sphere
                z = rng.uniform(r/2, r) # varied height off ground
                self.obstacles.append(Obstacle(
                    obstacle_type=ObstacleType.SPHERE,
                    position=np.array([cx, cy, z]),
                    dimensions=np.array([r*2, r*2, r*2]),
                    color=(0.4, 0.4, 0.4),
                    metadata={"layer": "boulder_sphere"}
                ))
            else:
                h = rng.uniform(r, r*2)
                self.obstacles.append(Obstacle(
                    obstacle_type=ObstacleType.CUBOID,
                    position=np.array([cx, cy, 0.0]),
                    dimensions=np.array([r*1.5, r*1.5, h]),
                    rotation_deg=rng.uniform(0, 360),
                    color=(0.35, 0.35, 0.35),
                    metadata={"layer": "boulder_box"}
                ))
            placed += 1
