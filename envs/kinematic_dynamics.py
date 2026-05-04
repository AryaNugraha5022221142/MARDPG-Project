import numpy as np

class KinematicDynamics:
    """
    Simplified kinematic model for UAV as described in the thesis.
    Ignores momentum. Agent outputs steering signals which directly update heading angles.
    """
    def __init__(self, dt: float = 0.01, v: float = 2.0):
        self.dt = dt
        self.v = v
        # State: [x, y, z, phi, theta, psi, vx, vy, vz, omega_x, omega_y, omega_z]
        # We maintain the 12-dim state shape so that ray_intersect and other env logic survives,
        # but we only actively update [0:3] and heading [4] (pitch=theta) / [5] (yaw=psi).
        self.state = np.zeros(12, dtype=np.float32)

    def reset(self, start_pos: np.ndarray, start_yaw: float = 0.0):
        self.state = np.zeros(12, dtype=np.float32)
        self.state[0:3] = start_pos
        self.state[5] = start_yaw  # psi (yaw, phi in prompt)
        self.state[4] = 0.0        # theta (pitch, vartheta in prompt)

    def rl_step(self, action: np.ndarray, lqr=None, M: int = 1):
        """
        action: [yaw_rate, pitch_rate] in rad/s.
        M: number of inner simulation ticks represented by one outer RL step.
        """
        yaw_rate = float(action[0])
        pitch_rate = float(action[1])
        control_dt = self.dt * M

        self.state[5] += yaw_rate * control_dt
        self.state[4] += pitch_rate * control_dt

        self.state[5] = (self.state[5] + np.pi) % (2 * np.pi) - np.pi
        self.state[4] = np.clip(self.state[4], -np.pi / 2, np.pi / 2)

        yaw = self.state[5]
        pitch = self.state[4]

        vx = self.v * np.cos(pitch) * np.cos(yaw)
        vy = self.v * np.cos(pitch) * np.sin(yaw)
        vz = self.v * np.sin(pitch)

        self.state[6] = vx
        self.state[7] = vy
        self.state[8] = vz

        self.state[0] += vx * control_dt
        self.state[1] += vy * control_dt
        self.state[2] += vz * control_dt

        self.state[10] = pitch_rate
        self.state[11] = yaw_rate

        return self.state.copy()
