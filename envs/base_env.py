"""
base_env.py
───────────
Shared data structures, configuration schema, collision utilities, and the
abstract BaseEnvironment class used by every scenario in this benchmark suite.

All coordinates are in metres; height/z is always the vertical axis.
"""

from __future__ import annotations

import abc
import dataclasses
import enum
import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class ObstacleType(enum.Enum):
    CUBOID    = "cuboid"
    CYLINDER  = "cylinder"
    IRREGULAR = "irregular"
    SPHERE    = "sphere"
    DYNAMIC   = "dynamic"


class DifficultyLevel(enum.Enum):
    TRIVIAL   = 1   # sanity-check; obstacles well separated
    EASY      = 2   # reference-paper baseline comparable
    MEDIUM    = 3   # denser, tighter passages
    HARD      = 4   # narrow corridors, local minima
    EXTREME   = 5   # near-impenetrable; stress-tests coordination


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Obstacle:
    """
    Unified obstacle representation for all environment types.

    Parameters
    ----------
    obstacle_type : ObstacleType
    position      : np.ndarray  shape (3,) – (x, y, z_base)
    dimensions    : np.ndarray  shape (3,) – (dx, dy, dz) for cuboids /
                                             (radius, radius, height) for cylinders
    rotation_deg  : float  – yaw around z-axis (cuboids only)
    color         : Tuple[float, float, float]  – RGB in [0, 1]
    is_dynamic    : bool
    velocity      : Optional[np.ndarray]  – (vx, vy, vz) m/s
    waypoints     : Optional[List[np.ndarray]]  – patrol path for dynamic obs.
    metadata      : dict  – arbitrary extra data (layer, cluster_id, …)
    """
    obstacle_type : ObstacleType
    position      : np.ndarray
    dimensions    : np.ndarray
    rotation_deg  : float = 0.0
    color         : Tuple[float, float, float] = (0.6, 0.6, 0.6)
    is_dynamic    : bool = False
    velocity      : Optional[np.ndarray] = None
    waypoints     : Optional[List[np.ndarray]] = None
    metadata      : dataclasses.field(default_factory=dict) = None

    def __post_init__(self):
        self.position   = np.asarray(self.position,   dtype=float)
        self.dimensions = np.asarray(self.dimensions, dtype=float)
        if self.metadata is None:
            self.metadata = {}

    # ── Bounding-box helpers ──────────────────────────────────────────────────

    @property
    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box (min_corner, max_corner) ignoring rotation."""
        half = self.dimensions / 2.0
        # For cylinders dimensions = (r, r, h); treat as square footprint
        lo = self.position - half
        hi = self.position + half
        lo[2] = self.position[2]                 # z_base
        hi[2] = self.position[2] + self.dimensions[2]
        return lo, hi

    def contains_point(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """
        Fast approximate point-in-obstacle test (ignores yaw rotation).
        """
        p = np.asarray(point, dtype=float)
        if self.obstacle_type in (ObstacleType.CYLINDER, ObstacleType.DYNAMIC):
            r = self.dimensions[0] + margin
            h = self.dimensions[2]
            dx = p[0] - self.position[0]
            dy = p[1] - self.position[1]
            dz = p[2] - self.position[2]
            return (dx*dx + dy*dy) <= r*r and 0 <= dz <= h
        elif self.obstacle_type == ObstacleType.SPHERE:
            rx, ry, rz = self.dimensions[0]/2 + margin, self.dimensions[1]/2 + margin, self.dimensions[2]/2 + margin
            center = np.array([self.position[0], self.position[1], self.position[2] + self.dimensions[2]/2])
            d = (p - center) / np.array([rx, ry, rz])
            return float(np.sum(d**2)) <= 1.0
        else:  # cuboid / irregular approximated as box
            lo, hi = self.aabb
            lo -= margin
            hi += margin
            return bool(np.all(p >= lo) and np.all(p <= hi))

    def clearance_to(self, point: np.ndarray) -> float:
        """Approximate signed distance from surface to point (+ = outside)."""
        p = np.asarray(point, dtype=float)
        if self.obstacle_type in (ObstacleType.CYLINDER, ObstacleType.DYNAMIC):
            r = self.dimensions[0]
            dx = p[0] - self.position[0]
            dy = p[1] - self.position[1]
            return math.sqrt(dx*dx + dy*dy) - r
        elif self.obstacle_type == ObstacleType.SPHERE:
            center = np.array([self.position[0], self.position[1], self.position[2] + self.dimensions[2]/2])
            # For simplicity, approximate distance using max dimension radius
            r = max(self.dimensions) / 2.0
            return float(np.linalg.norm(p - center)) - r
        lo, hi = self.aabb
        centre = (lo + hi) / 2.0
        half   = (hi - lo) / 2.0
        d = np.abs(p - centre) - half
        return float(np.max(d))   # negative inside, positive outside


@dataclasses.dataclass
class EnvironmentConfig:
    """
    Full specification for a MARDPG benchmark environment.

    Required by every environment constructor; serves as a canonical record of
    every hyperparameter so experiments are reproducible from config alone.
    """
    # ── Map ──────────────────────────────────────────────────────────────────
    map_width         : float = 50.0    # metres, x-axis
    map_depth         : float = 50.0    # metres, y-axis
    map_height        : float = 20.0    # metres, z-axis ceiling

    # ── Obstacles ─────────────────────────────────────────────────────────────
    obstacle_density  : float = 0.15    # fraction of map area covered by obstacles
    min_height        : float = 2.0
    max_height        : float = 10.0
    height_dist       : str   = "uniform"   # "uniform" | "normal" | "exponential"
    min_clearance     : float = 2.5     # minimum gap between any two obstacles (m)
    # Per-type overrides (used by mixed environments)
    type_ratios       : Dict[str, float] = dataclasses.field(
        default_factory=lambda: {"cuboid": 1.0})

    # ── Agents ───────────────────────────────────────────────────────────────
    n_agents          : int   = 4
    agent_radius      : float = 0.3     # m – used for collision checks
    uav_spawn_mode    : str   = "corners"  # "corners"|"random"|"clustered"|"border"
    goal_mode         : str   = "opposite" # "opposite"|"random"|"fixed"

    # ── Physics / sensors ────────────────────────────────────────────────────
    collision_threshold : float = 0.5   # m – agent-obstacle collision radius
    agent_agent_threshold: float = 0.6  # m – agent-agent collision radius
    sensor_range      : float = 10.0    # m – partial observability radius
    max_speed         : float = 3.0     # m/s

    # ── Scenario meta ─────────────────────────────────────────────────────────
    difficulty        : DifficultyLevel = DifficultyLevel.MEDIUM
    seed              : int  = 42
    name              : str  = "unnamed"
    description       : str  = ""

    def summary(self) -> str:
        lines = [
            f"{'─'*60}",
            f"  Environment : {self.name}",
            f"  Difficulty  : {self.difficulty.name}  (level {self.difficulty.value})",
            f"  Map         : {self.map_width}m × {self.map_depth}m × {self.map_height}m",
            f"  Density     : {self.obstacle_density:.2%}",
            f"  Heights     : [{self.min_height}, {self.max_height}] m  ({self.height_dist})",
            f"  Min gap     : {self.min_clearance} m",
            f"  Agents      : {self.n_agents}  (spawn={self.uav_spawn_mode})",
            f"  Collision   : agent-obs={self.collision_threshold}m  "
                           f"agent-agent={self.agent_agent_threshold}m",
            f"  Sensor rng  : {self.sensor_range} m",
            f"  Seed        : {self.seed}",
            f"  Description : {self.description}",
            f"{'─'*60}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

class _PlacementGrid:
    """
    Lightweight 2-D occupancy grid that accelerates obstacle placement by
    rejecting positions that violate the minimum-clearance constraint without
    iterating over all existing obstacles.
    """
    def __init__(self, width: float, depth: float, cell: float = 1.0):
        self.width  = width
        self.depth  = depth
        self.cell   = cell
        self.cols   = math.ceil(width / cell)
        self.rows   = math.ceil(depth / cell)
        self._grid  = np.zeros((self.rows, self.cols), dtype=bool)

    def _cells_for_circle(self, x: float, y: float, r: float):
        c0 = max(0, int((x - r) / self.cell))
        c1 = min(self.cols - 1, int((x + r) / self.cell))
        r0 = max(0, int((y - r) / self.cell))
        r1 = min(self.rows - 1, int((y + r) / self.cell))
        return c0, c1, r0, r1

    def is_free(self, x: float, y: float, r: float) -> bool:
        c0, c1, r0, r1 = self._cells_for_circle(x, y, r)
        return not self._grid[r0:r1+1, c0:c1+1].any()

    def mark(self, x: float, y: float, r: float):
        c0, c1, r0, r1 = self._cells_for_circle(x, y, r)
        self._grid[r0:r1+1, c0:c1+1] = True


def sample_height(rng: np.random.Generator,
                  cfg: EnvironmentConfig,
                  n: int = 1) -> np.ndarray:
    """Draw obstacle heights from the configured distribution."""
    lo, hi = cfg.min_height, cfg.max_height
    if cfg.height_dist == "uniform":
        return rng.uniform(lo, hi, n)
    elif cfg.height_dist == "normal":
        mu  = (lo + hi) / 2.0
        sig = (hi - lo) / 6.0
        return np.clip(rng.normal(mu, sig, n), lo, hi)
    elif cfg.height_dist == "exponential":
        vals = rng.exponential(scale=(hi - lo) / 3.0, size=n) + lo
        return np.clip(vals, lo, hi)
    elif cfg.height_dist == "bimodal":
        # mix of short & tall buildings
        low_h  = rng.uniform(lo, (lo + hi) / 2, n)
        high_h = rng.uniform((lo + hi) / 2, hi, n)
        mask   = rng.random(n) > 0.5
        return np.where(mask, high_h, low_h)
    else:
        raise ValueError(f"Unknown height distribution: {cfg.height_dist}")


def random_color(rng: np.random.Generator) -> Tuple[float, float, float]:
    hue = rng.uniform(0, 1)
    return tuple(float(x) for x in _hsv_to_rgb(hue, 0.8, 0.9))


def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    i = int(h * 6)
    f = h * 6 - i
    p, q, t = v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
    r, g, b = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i % 6]
    return np.array([r, g, b])


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base environment
# ──────────────────────────────────────────────────────────────────────────────

class BaseEnvironment(abc.ABC):
    """
    Abstract base class for all MARDPG benchmark environments.

    Subclasses implement `_generate_obstacles()` which populates
    `self.obstacles`.  Everything else – agent spawn, goal placement,
    validation, collision checks, and export – is handled here.
    """

    def __init__(self, config: EnvironmentConfig):
        self.config    = config
        self.rng       = np.random.default_rng(config.seed)
        self.obstacles : List[Obstacle] = []
        self.agents    : List[np.ndarray] = []   # shape (3,) positions
        self.goals     : List[np.ndarray] = []   # shape (3,) positions
        self._t        : float = 0.0             # simulation time (for dynamic obs.)

        random.seed(config.seed)
        self._build()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> Dict:
        """Reset agents, goals, and dynamic obstacles to initial state."""
        self._t      = 0.0
        self.agents  = self._place_agents()
        self.goals   = self._place_goals()
        return self._obs_dict()

    def step(self, actions: List[np.ndarray], dt: float = 0.1) -> Tuple[Dict, List[float], List[bool], Dict]:
        """
        Advance the environment by one time step.

        Parameters
        ----------
        actions : list of np.ndarray  shape (3,) – desired velocity vectors
        dt      : float – time step in seconds

        Returns
        -------
        obs     : dict of per-agent observations
        rewards : list of per-agent scalar rewards
        dones   : list of per-agent done flags
        info    : diagnostic dict
        """
        self._t += dt
        self._update_dynamic_obstacles(dt)
        self._apply_actions(actions, dt)
        rewards = self._compute_rewards()
        dones   = self._compute_dones()
        info    = {"t": self._t, "n_obs": len(self.obstacles)}
        return self._obs_dict(), rewards, dones, info

    def is_collision(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Return True if `point` collides with any obstacle."""
        thr = self.config.collision_threshold + margin
        for obs in self.obstacles:
            if obs.contains_point(point, margin=thr):
                return True
        return False

    def min_clearance_at(self, point: np.ndarray) -> float:
        """Minimum clearance from `point` to the nearest obstacle surface."""
        if not self.obstacles:
            return float("inf")
        return min(obs.clearance_to(point) for obs in self.obstacles)

    def validate(self) -> Dict[str, object]:
        """
        Run a suite of environment-health checks.
        Returns a dict of metrics and any warnings discovered.
        """
        warnings = []
        metrics  = {}

        # Obstacle-obstacle clearance
        min_gap = float("inf")
        
        def get_radius(obs):
            if obs.obstacle_type in (ObstacleType.CYLINDER, ObstacleType.DYNAMIC):
                return obs.dimensions[0]
            elif obs.obstacle_type == ObstacleType.SPHERE:
                return max(obs.dimensions) / 2.0
            else:
                return max(obs.dimensions[:2]) / 2.0
                
        for i, a in enumerate(self.obstacles):
            ra = get_radius(a)
            for b in self.obstacles[i+1:]:
                if a.metadata.get("allow_overlap") or b.metadata.get("allow_overlap"):
                    continue
                rb = get_radius(b)
                d = np.linalg.norm(a.position[:2] - b.position[:2])
                gap = d - (ra + rb)
                min_gap = min(min_gap, gap)
        metrics["min_obstacle_gap_m"] = round(min_gap, 3)
        if min_gap < -0.1:
            warnings.append(f"WARNING: Obstacles overlap! min gap = {min_gap:.2f} m")
        elif min_gap < 0.8:
            warnings.append(f"WARNING: min gap {min_gap:.2f} m is very narrow (< 0.8 m)")

        # Agent start/goal reachability (simple free-space check)
        for i, (ag, gl) in enumerate(zip(self.agents, self.goals)):
            if self.is_collision(ag):
                warnings.append(f"WARNING: Agent {i} spawned inside an obstacle!")
            if self.is_collision(gl):
                warnings.append(f"WARNING: Goal  {i} placed inside an obstacle!")

        # Coverage
        area = self.config.map_width * self.config.map_depth
        covered = sum(o.dimensions[0] * o.dimensions[1] for o in self.obstacles)
        metrics["obstacle_coverage_pct"] = round(100 * covered / area, 2)
        metrics["n_obstacles"]           = len(self.obstacles)
        metrics["n_dynamic"]             = sum(1 for o in self.obstacles if o.is_dynamic)

        metrics["warnings"] = warnings
        return metrics

    def to_dict(self) -> Dict:
        """Serialise environment to a plain Python dict (JSON-compatible)."""
        def _obs(o: Obstacle):
            return {
                "type"      : o.obstacle_type.value,
                "position"  : o.position.tolist(),
                "dimensions": o.dimensions.tolist(),
                "rotation"  : o.rotation_deg,
                "is_dynamic": o.is_dynamic,
                "velocity"  : o.velocity.tolist() if o.velocity is not None else None,
                "color"     : list(o.color),
                "metadata"  : o.metadata,
            }
        return {
            "config"   : dataclasses.asdict(self.config),
            "obstacles": [_obs(o) for o in self.obstacles],
            "agents"   : [a.tolist() for a in self.agents],
            "goals"    : [g.tolist() for g in self.goals],
        }

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def _generate_obstacles(self):
        """Populate self.obstacles.  Called once during __init__."""

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build(self):
        self._generate_obstacles()
        self.agents = self._place_agents()
        self.goals  = self._place_goals()

    def _free_position_2d(self, margin: float, max_tries: int = 2000) -> Optional[np.ndarray]:
        """Sample a random 2-D position that is collision-free."""
        cfg = self.config
        for _ in range(max_tries):
            x = self.rng.uniform(margin, cfg.map_width  - margin)
            y = self.rng.uniform(margin, cfg.map_depth  - margin)
            p = np.array([x, y, 1.5])   # UAV cruise altitude
            if not self.is_collision(p, margin=0.0):
                return p
        return None

    def _place_agents(self) -> List[np.ndarray]:
        cfg = self.config
        n   = cfg.n_agents
        mode = cfg.uav_spawn_mode
        margin = cfg.collision_threshold + cfg.agent_radius + 1.0
        positions = []

        def valid_pos(p):
            if self.is_collision(p, margin=1.0): return False
            for prev_p in positions:
                if np.linalg.norm(p - prev_p) < (cfg.agent_radius * 2 + 1.0):
                    return False
            return True

        if mode == "corners":
            corners = [
                np.array([margin, margin, 1.5]),
                np.array([cfg.map_width - margin, margin, 1.5]),
                np.array([margin, cfg.map_depth - margin, 1.5]),
                np.array([cfg.map_width - margin, cfg.map_depth - margin, 1.5]),
            ]
            for i in range(n):
                if i < len(corners):
                    c = corners[i].copy()
                else:
                    p = self._free_position_2d(margin)
                    c = p if p is not None else corners[i % len(corners)].copy()
                if not valid_pos(c):
                    # Try to find a free space nearby
                    found = False
                    for radius in np.arange(1.0, 10.0, 0.5):
                        for angle in np.linspace(0, 2 * np.pi, 16):
                            test_c = c + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
                            test_c[0] = np.clip(test_c[0], margin, cfg.map_width - margin)
                            test_c[1] = np.clip(test_c[1], margin, cfg.map_depth - margin)
                            if valid_pos(test_c):
                                c = test_c
                                found = True
                                break
                        if found: break
                positions.append(c.copy())
        elif mode == "border":
            for i in range(n):
                side = i % 4
                if side == 0:
                    p = np.array([self.rng.uniform(margin, cfg.map_width - margin), margin, 1.5])
                elif side == 1:
                    p = np.array([cfg.map_width - margin, self.rng.uniform(margin, cfg.map_depth - margin), 1.5])
                elif side == 2:
                    p = np.array([self.rng.uniform(margin, cfg.map_width - margin), cfg.map_depth - margin, 1.5])
                else:
                    p = np.array([margin, self.rng.uniform(margin, cfg.map_depth - margin), 1.5])
                positions.append(p)
        elif mode == "clustered":
            cx = self.rng.uniform(margin * 2, cfg.map_width  - margin * 2)
            cy = self.rng.uniform(margin * 2, cfg.map_depth  - margin * 2)
            for _ in range(n):
                p = self._free_position_2d(margin)
                if p is None: p = np.array([cx, cy, 1.5])
                p[0] = np.clip(cx + self.rng.normal(0, 2.0), margin, cfg.map_width  - margin)
                p[1] = np.clip(cy + self.rng.normal(0, 2.0), margin, cfg.map_depth  - margin)
                positions.append(p)
        else:  # "random"
            for _ in range(n):
                p = self._free_position_2d(margin)
                if p is None: p = np.array([margin, margin, 1.5])
                positions.append(p)

        return positions

    def _place_goals(self) -> List[np.ndarray]:
        cfg    = self.config
        margin = cfg.collision_threshold + cfg.agent_radius + 1.0
        goals  = []
        mode   = cfg.goal_mode

        for i, agent in enumerate(self.agents):
            if mode == "opposite":
                # Mirror agent position across map centre
                cx = cfg.map_width  / 2.0
                cy = cfg.map_depth  / 2.0
                gx = np.clip(2 * cx - agent[0], margin, cfg.map_width  - margin)
                gy = np.clip(2 * cy - agent[1], margin, cfg.map_depth  - margin)
                g = np.array([gx, gy, 1.5])
                if self.is_collision(g):
                    found = False
                    for radius in np.arange(1.0, 8.0, 0.5):
                        for angle in np.linspace(0, 2 * np.pi, 8):
                            test_g = g + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
                            test_g[0] = np.clip(test_g[0], margin, cfg.map_width - margin)
                            test_g[1] = np.clip(test_g[1], margin, cfg.map_depth - margin)
                            if not self.is_collision(test_g):
                                g = test_g
                                found = True
                                break
                        if found: break
                if self.is_collision(g):
                    fallback = self._free_position_2d(margin)
                    if fallback is not None: g = fallback
            elif mode == "random":
                g = self._free_position_2d(margin)
                if g is None: g = np.array([cfg.map_width - margin, cfg.map_depth - margin, 1.5])
            else:  # "fixed"
                g = np.array([cfg.map_width - margin, cfg.map_depth - margin, 1.5])
            goals.append(g)
        return goals

    def _update_dynamic_obstacles(self, dt: float):
        for obs in self.obstacles:
            if not obs.is_dynamic or obs.velocity is None:
                continue
            if obs.waypoints:
                self._patrol_step(obs, dt)
            else:
                # Bounce off walls
                obs.position += obs.velocity * dt
                cfg = self.config
                for ax, lim in zip([0, 1], [cfg.map_width, cfg.map_depth]):
                    if obs.position[ax] < 0 or obs.position[ax] > lim:
                        obs.velocity[ax] *= -1
                        obs.position[ax] = np.clip(obs.position[ax], 0, lim)

    def _patrol_step(self, obs: Obstacle, dt: float):
        if "wp_idx" not in obs.metadata:
            obs.metadata["wp_idx"] = 0
        idx    = obs.metadata["wp_idx"]
        target = obs.waypoints[idx]
        diff   = target - obs.position
        dist   = np.linalg.norm(diff)
        speed  = np.linalg.norm(obs.velocity)
        if dist < speed * dt:
            obs.position = target.copy()
            obs.metadata["wp_idx"] = (idx + 1) % len(obs.waypoints)
        else:
            obs.position += (diff / dist) * speed * dt

    def _apply_actions(self, actions: List[np.ndarray], dt: float):
        cfg  = self.config
        vmax = cfg.max_speed
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            vel = np.asarray(action, dtype=float)
            spd = np.linalg.norm(vel)
            if spd > vmax:
                vel = vel / spd * vmax
            new_pos = agent + vel * dt
            new_pos[0] = np.clip(new_pos[0], 0, cfg.map_width)
            new_pos[1] = np.clip(new_pos[1], 0, cfg.map_depth)
            new_pos[2] = np.clip(new_pos[2], 0, cfg.map_height)
            if not self.is_collision(new_pos):
                self.agents[i] = new_pos

    def _compute_rewards(self) -> List[float]:
        rewards = []
        for agent, goal in zip(self.agents, self.goals):
            dist    = float(np.linalg.norm(agent - goal))
            coll    = self.is_collision(agent)
            r       = -dist * 0.1 + (-5.0 if coll else 0.0)
            rewards.append(r)
        return rewards

    def _compute_dones(self) -> List[bool]:
        dones = []
        for agent, goal in zip(self.agents, self.goals):
            dist  = float(np.linalg.norm(agent - goal))
            coll  = self.is_collision(agent)
            dones.append(dist < self.config.collision_threshold or coll)
        return dones

    def _obs_dict(self) -> Dict:
        obs = {}
        sr  = self.config.sensor_range
        for i, agent in enumerate(self.agents):
            # Gather nearby obstacles within sensor range
            nearby = [
                {"pos": o.position.tolist(), "dim": o.dimensions.tolist(),
                 "type": o.obstacle_type.value}
                for o in self.obstacles
                if np.linalg.norm(o.position - agent) < sr
            ]
            # Other agents within sensor range
            others = [
                self.agents[j].tolist()
                for j in range(len(self.agents))
                if j != i and np.linalg.norm(self.agents[j] - agent) < sr
            ]
            obs[f"agent_{i}"] = {
                "self_pos"         : agent.tolist(),
                "goal_pos"         : self.goals[i].tolist(),
                "relative_goal"    : (self.goals[i] - agent).tolist(),
                "nearby_obstacles" : nearby,
                "nearby_agents"    : others,
            }
        return obs

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} "
                f"name={self.config.name!r} "
                f"n_obs={len(self.obstacles)} "
                f"difficulty={self.config.difficulty.name}>")
