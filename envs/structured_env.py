"""
structured_env.py  –  Structured Periodic Environment Analogue
──────────────────────────────────────────────────────────────
Highly predictable, matrix-like alignments like warehouses or arrays.
"""

from __future__ import annotations

import math
import numpy as np

from .base_env import BaseEnvironment, EnvironmentConfig, Obstacle, ObstacleType

class StructuredPeriodicEnvironment(BaseEnvironment):
    """
    Warehouse shelves, server racks, solar panel arrays, etc.
    """
    def _generate_obstacles(self):
        cfg = self.config
        W, D = cfg.map_width, cfg.map_depth
        
        # Hardcode based on difficulty
        diff_val = cfg.difficulty.value
        n_rows = 3 + diff_val * 2
        n_cols = 3 + diff_val * 2
        
        spacing_x = W / (n_cols + 1)
        spacing_y = D / (n_rows + 1)
        
        width = min(2.0, spacing_x - cfg.min_clearance)
        depth = min(8.0, spacing_y * 0.8)
        
        h = cfg.max_height * 0.7
        
        for r in range(n_rows):
            for c in range(n_cols):
                cx = spacing_x * (c + 1)
                cy = spacing_y * (r + 1)
                
                # occasional missing rack
                if self.rng.random() < 0.1 * diff_val:
                    continue
                    
                self.obstacles.append(Obstacle(
                    obstacle_type=ObstacleType.CUBOID,
                    position=np.array([cx, cy, 0.0]),
                    dimensions=np.array([width, depth, h]),
                    color=(0.2, 0.6, 0.8),
                    metadata={"layer": "rack"}
                ))
