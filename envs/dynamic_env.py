"""
dynamic_env.py  –  Dynamic Obstacle Environment
───────────────────────────────────────────────
Populates the environment with moving obstacles (pedestrians, other vehicles, debris).
"""

from __future__ import annotations

import math
import numpy as np
from typing import List

from .base_env import BaseEnvironment, EnvironmentConfig, Obstacle, ObstacleType, _PlacementGrid

class DynamicObstacleEnvironment(BaseEnvironment):
    def _generate_obstacles(self):
        cfg = self.config
        W, D = cfg.map_width, cfg.map_depth
        diff_val = cfg.difficulty.value
        
        n_static = 10 * diff_val
        n_dynamic = 5 * diff_val
        
        pgrid = _PlacementGrid(W, D, cell=1.0)
        
        # Add a few static obstacles
        for _ in range(n_static):
            cx = self.rng.uniform(2.0, W - 2.0)
            cy = self.rng.uniform(2.0, D - 2.0)
            r = self.rng.uniform(1.0, 3.0)
            if pgrid.is_free(cx, cy, r + cfg.min_clearance):
                pgrid.mark(cx, cy, r)
                self.obstacles.append(Obstacle(
                    obstacle_type=ObstacleType.CYLINDER,
                    position=np.array([cx, cy, 0.0]),
                    dimensions=np.array([r, r, cfg.max_height * 0.5]),
                    color=(0.5, 0.5, 0.5)
                ))
                
        # Add dynamic obstacles
        for _ in range(n_dynamic):
            cx = self.rng.uniform(2.0, W - 2.0)
            cy = self.rng.uniform(2.0, D - 2.0)
            r = self.rng.uniform(0.5, 1.5)
            
            vx = self.rng.uniform(-2.0, 2.0)
            vy = self.rng.uniform(-2.0, 2.0)
            
            self.obstacles.append(Obstacle(
                obstacle_type=ObstacleType.DYNAMIC,
                position=np.array([cx, cy, 1.0]),
                dimensions=np.array([r, r, r*2]),
                color=(1.0, 0.5, 0.0),
                is_dynamic=True,
                velocity=np.array([vx, vy, 0.0]),
                metadata={"layer": "dynamic"}
            ))
