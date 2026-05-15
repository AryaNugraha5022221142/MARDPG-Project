"""
mixed_env.py  –  Mixed Scenario Environment
─────────────────────────────────────────────
Combines urban, forest, and structured generation logic into zones.
"""

from __future__ import annotations

import math
import numpy as np

from .base_env import BaseEnvironment, EnvironmentConfig, Obstacle, ObstacleType
from .urban_env import DenseUrbanEnvironment
from .forest_env import CylindricalForestEnvironment
from .structured_env import StructuredPeriodicEnvironment

class MixedObstacleEnvironment(BaseEnvironment):
    def _generate_obstacles(self):
        # We will manually merge logic from other generators in different quadrants
        cfg = self.config
        W, D = cfg.map_width, cfg.map_depth
        
        # Quarter sizes
        qw, qd = W/2, D/2
        
        # Q1: Urban
        sub_cfg1 = EnvironmentConfig(**{**cfg.__dict__, "map_width": qw, "map_depth": qd})
        urban = DenseUrbanEnvironment(sub_cfg1)
        # Shift urban obstacles
        for o in urban.obstacles:
            o.position[0] += qw
            o.position[1] += qd
            self.obstacles.append(o)
            
        # Q2: Forest
        sub_cfg2 = EnvironmentConfig(**{**cfg.__dict__, "map_width": qw, "map_depth": qd})
        forest = CylindricalForestEnvironment(sub_cfg2)
        for o in forest.obstacles:
           # Q2 is x<qw, y>qd
           o.position[1] += qd
           self.obstacles.append(o)
           
        # Q3: Structured
        sub_cfg3 = EnvironmentConfig(**{**cfg.__dict__, "map_width": qw, "map_depth": qd})
        struct = StructuredPeriodicEnvironment(sub_cfg3)
        for o in struct.obstacles:
            # Q3 is x<qw, y<qd
            self.obstacles.append(o)
            
        # Q4 is empty / central plaza
