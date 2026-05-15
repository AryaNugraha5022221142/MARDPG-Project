"""
benchmark_suite.py
──────────────────
Entry point for creating and cataloging all MARDPG benchmark environments.
"""

from .base_env import EnvironmentConfig, DifficultyLevel
from .urban_env import DenseUrbanEnvironment
from .forest_env import CylindricalForestEnvironment
from .terrain_env import IrregularTerrainEnvironment
from .structured_env import StructuredPeriodicEnvironment
from .mixed_env import MixedObstacleEnvironment
from .dynamic_env import DynamicObstacleEnvironment

class BenchmarkSuite:
    """Registry and factory for benchmark environments."""
    
    REGISTRY = {
        "urban": DenseUrbanEnvironment,
        "forest": CylindricalForestEnvironment,
        "terrain": IrregularTerrainEnvironment,
        "structured": StructuredPeriodicEnvironment,
        "mixed": MixedObstacleEnvironment,
        "dynamic": DynamicObstacleEnvironment,
    }
    
    @classmethod
    def make(cls, name: str, config: EnvironmentConfig):
        if name not in cls.REGISTRY:
            raise ValueError(f"Unknown benchmark environment: {name}")
        return cls.REGISTRY[name](config)

def create_all_benchmarks(difficulty=DifficultyLevel.MEDIUM):
    envs = {}
    base_cfg = EnvironmentConfig(map_width=60.0, map_depth=60.0, difficulty=difficulty)
    for name in BenchmarkSuite.REGISTRY.keys():
        cfg = EnvironmentConfig(**base_cfg.__dict__)
        cfg.name = name
        envs[name] = BenchmarkSuite.make(name, cfg)
    return envs
