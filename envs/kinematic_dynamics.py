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
        action: [p, tau] (steering signals)
        p: yaw steering (changes phi/psi)
        tau: pitch steering (changes theta/vartheta)
        """
        p = action[0]
        tau = action[1]
        
        # In the context of MARDPG kinematic model, steering signals update angles directly.
        # It's an end-to-end update where the signals are directly added per timestep.
        # Or, they act as rate commands. Given the text: "These signals are added directly
        # to the current heading angles to determine the UAVs position in the next time step"
        
        # New heading angles:
        self.state[5] += p # yaw (psi)
        self.state[4] += tau # pitch (theta)
        
        yaw = self.state[5]
        pitch = self.state[4]
        
        # Position update based on constant velocity and new heading:
        # v_x = v * cos(pitch) * cos(yaw)
        # v_y = v * cos(pitch) * sin(yaw)
        # v_z = v * sin(pitch)
        
        vx = self.v * np.cos(pitch) * np.cos(yaw)
        vy = self.v * np.cos(pitch) * np.sin(yaw)
        vz = self.v * np.sin(pitch)
        
        self.state[6] = vx
        self.state[7] = vy
        self.state[8] = vz
        
        # Update position
        self.state[0] += vx * self.dt
        self.state[1] += vy * self.dt
        self.state[2] += vz * self.dt
        
        # Wrap angles to [-pi, pi]
        self.state[5] = (self.state[5] + np.pi) % (2 * np.pi) - np.pi
        self.state[4] = np.clip(self.state[4], -np.pi/2, np.pi/2)
        
        return self.state.copy()
