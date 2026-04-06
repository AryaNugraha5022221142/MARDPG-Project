# envs/dynamics.py
import numpy as np

class QuadcopterDynamics:
    """
    First-order velocity tracking dynamics for a quadcopter.
    """
    def __init__(self, dt: float = 0.01, tau: float = 0.1, noise_std: float = 0.0):
        self.dt = dt
        self.tau = tau
        self.noise_std = noise_std
        
        # State: [x, y, z, phi, theta, psi, vx, vy, vz, omega_x, omega_y, omega_z]
        self.state = np.zeros(12, dtype=np.float32)

    def reset(self, start_pos: np.ndarray, start_yaw: float = 0.0):
        """Resets the state to a given position and yaw."""
        self.state = np.zeros(12, dtype=np.float32)
        self.state[0:3] = start_pos
        self.state[5] = start_yaw

    def rl_step(self, velocity_ref: np.ndarray, lqr, M: int = 10):
        """
        Outer RL step: runs M inner LQR steps.
        velocity_ref: [vx*, vy*, vz*, yaw_rate*] from RL policy
        """
        yaw_rate_cmd_rad = np.deg2rad(velocity_ref[3])
        
        for _ in range(M):
            for j, axis in enumerate([0, 1, 2]):
                # Eq. 3.68: Reference generation
                p_ref = self.state[axis] + velocity_ref[j] * self.dt
                v_ref = velocity_ref[j]
                
                # Eq. 3.69: LQR feedback + feedforward
                u_j = lqr.compute_input(p_ref, v_ref, self.state[axis], self.state[6+j])
                u_j = np.clip(u_j, -5.0, 5.0)  # Actuator saturation Eq. 3.74
                
                # Apply u_j to plant dynamics (ZOH update, Eq. 3.16)
                alpha = np.exp(-self.dt / self.tau)
                self.state[6+j] = alpha * self.state[6+j] + (1 - alpha) * u_j
                self.state[axis] += self.state[6+j] * self.dt
            
            # Yaw update (inner loop)
            psi = self.state[5]
            psi_new = psi + yaw_rate_cmd_rad * self.dt
            self.state[5] = (psi_new + np.pi) % (2 * np.pi) - np.pi
            
        # Roll and pitch estimation (simplified attitude tracking)
        vx_cmd, vy_cmd, vz_cmd = velocity_ref[0:3]
        self.state[3] = np.arctan2(vy_cmd, np.sqrt(vx_cmd**2 + 0.01)) * 0.5
        self.state[4] = np.arctan2(vz_cmd, np.sqrt(vx_cmd**2 + vy_cmd**2 + 0.01)) * 0.5
        
        return self.state.copy()

    def step(self, velocity_cmd: np.ndarray):
        """
        Updates the quadcopter state based on velocity commands.
        velocity_cmd: [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd (deg/s)]
        """
        v_cmd = velocity_cmd[0:3]
        # Add disturbance noise if configured
        if self.noise_std > 0:
            v_cmd = v_cmd + np.random.normal(0, self.noise_std, size=3)
            
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
