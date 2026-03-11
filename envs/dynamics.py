# envs/dynamics.py
import numpy as np

class QuadcopterDynamics:
    """
    First-order velocity tracking dynamics for a quadcopter.
    """
    def __init__(self, dt: float = 0.01, tau: float = 0.1):
        self.dt = dt
        self.tau = tau
        
        # State: [x, y, z, phi, theta, psi, vx, vy, vz, omega_x, omega_y, omega_z]
        self.state = np.zeros(12, dtype=np.float32)

    def reset(self, start_pos: np.ndarray, start_yaw: float = 0.0):
        """Resets the state to a given position and yaw."""
        self.state = np.zeros(12, dtype=np.float32)
        self.state[0:3] = start_pos
        self.state[5] = start_yaw

    def step(self, velocity_cmd: np.ndarray):
        """
        Updates the quadcopter state based on velocity commands.
        velocity_cmd: [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd (deg/s)]
        """
        v_cmd = velocity_cmd[0:3]
        yaw_rate_cmd_rad = np.deg2rad(velocity_cmd[3])
        
        # Current velocities
        v = self.state[6:9]
        
        # Velocity update
        v_new = v + (v_cmd - v) / self.tau * self.dt
        
        # Position update (Euler integration)
        x_new = self.state[0:3] + v_new * self.dt
        
        # Yaw update
        psi = self.state[5]
        psi_new = psi + yaw_rate_cmd_rad * self.dt
        # Wrap to [-pi, pi]
        psi_new = (psi_new + np.pi) % (2 * np.pi) - np.pi
        
        # Roll and pitch estimation
        vx_cmd, vy_cmd, vz_cmd = v_cmd
        phi_new = np.arctan2(vy_cmd, np.sqrt(vx_cmd**2 + 0.01)) * 0.5
        theta_new = np.arctan2(vz_cmd, np.sqrt(vx_cmd**2 + vy_cmd**2 + 0.01)) * 0.5
        
        # Update state
        self.state[0:3] = x_new
        self.state[3] = phi_new
        self.state[4] = theta_new
        self.state[5] = psi_new
        self.state[6:9] = v_new
        self.state[9:12] = 0.0  # Simplified angular velocities
        
        return self.state.copy()
