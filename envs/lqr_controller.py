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

        # Cost matrices (tune these — see methodology Section 3.11.4)
        self.Q_lqr = np.diag([10.0, 1.0]) if Q is None else Q  # penalise pos error more
        self.R_lqr = np.array([[0.1]]) if R is None else R

        # Solve DARE for optimal gain
        P = solve_discrete_are(self.A, self.B, self.Q_lqr, self.R_lqr)
        self.K = np.linalg.inv(self.R_lqr + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        # K shape: (1, 2) — gain on [pos_error, vel_error]

    def compute_input(self, p_ref, v_ref, p, v):
        """
        Feedback-corrected reference tracking (Eq. 3.68–3.69).
        p_ref, v_ref: reference position and velocity from RL action
        p, v: current position and velocity
        Returns: u (scalar velocity command to plant)
        """
        e = np.array([p - p_ref, v - v_ref])  # x - x_ref (thesis sign)
        u = v_ref - self.K @ e               # LQR feedback + feedforward
        return float(u)
