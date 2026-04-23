# envs/lqr_controller.py
import numpy as np
from scipy.linalg import solve_discrete_are

class PerAxisLQR:
    """
    Per-axis LQR tracking controller.
    State: [p_j, v_j], Control: u_j (velocity command to plant).
    Methodology Eq. 3.13, 3.54–3.60.
    """
    def __init__(self, tau=0.1, dt=0.01, Q=None, R=None):
        self.dt = dt
        self.tau = tau
        alpha = np.exp(-dt / tau)  # Eq. 3.16: ZOH decay coeff

        beta = dt - tau * (1 - alpha)
        self.A = np.array([[1.0, tau * (1 - alpha)],
                           [0.0, alpha]])
        self.B = np.array([[beta],
                           [1 - alpha]])

        # Cost matrices tuned to keep k_v < 0.5 and prevent actuator saturation (>5.0 m/s clipping)
        # Prevents the actor gradient from dropping to zero at high velocities
        self.Q_lqr = np.diag([1.0, 0.1]) if Q is None else Q
        self.R_lqr = np.array([[2.0]]) if R is None else R

        # Solve DARE for optimal gain
        P = solve_discrete_are(self.A, self.B, self.Q_lqr, self.R_lqr)
        self.K = np.linalg.inv(self.R_lqr + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        self.k_v = self.K[0, 1]  # velocity gain only
        
        Acl = self.A - self.B @ self.K
        assert np.max(np.abs(np.linalg.eigvals(Acl))) < 1.0, 'LQR unstable'

    def compute_velocity_input(self, v_ref, v):
        """
        Feedback-corrected reference tracking (pure velocity LQR).
        v_ref: reference velocity from RL action
        v: current velocity
        Returns: u (scalar velocity command to plant)
        """
        u = v_ref + self.k_v * (v_ref - v)
        return float(u)
