import numpy as np

class KinematicDynamics:
    """
    Simplified kinematic model for UAV as described in the thesis.
    Ignores momentum. Agent outputs steering signals which directly update heading angles.
    """
    def __init__(self, dt: float = 0.01, v: float = 2.0):
        self.dt = dt
        self.v = v
        # State: [x, y, z, roll(phi), pitch(theta), yaw(psi), vx, vy, vz, p, q, r]
        # Paper mapping: vartheta = horizontal/yaw angle -> state[5] (psi)
        # Paper mapping: varphi = vertical/elevation angle -> state[4] (theta)
        self.state = np.zeros(12, dtype=np.float32)

    def reset(self, start_pos: np.ndarray, start_yaw: float = 0.0):
        self.state = np.zeros(12, dtype=np.float32)
        self.state[0:3] = start_pos
        self.state[5] = start_yaw  # psi/yaw = paper vartheta
        self.state[4] = 0.0        # theta/pitch = paper varphi
        self.state[6] = self.v * np.cos(self.state[4]) * np.cos(self.state[5])
        self.state[7] = self.v * np.cos(self.state[4]) * np.sin(self.state[5])
        self.state[8] = self.v * np.sin(self.state[4])
        return self.state.copy()

    def rl_step(self, action: np.ndarray, M: int = 1):
        """
        action: [rho, tau] direct steering angle increments.
        rho updates horizontal/yaw angle; tau updates vertical/pitch angle.
        M is the number of inner steps.
        """
        rho_step = float(action[0]) / M
        tau_step = float(action[1]) / M

        for _ in range(M):
            self.state[5] += rho_step
            self.state[4] += tau_step

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

            self.state[0] += vx * self.dt
            self.state[1] += vy * self.dt
            self.state[2] += vz * self.dt

        self.state[9] = 0.0
        self.state[10] = tau_step * M
        self.state[11] = rho_step * M

        return self.state.copy()
