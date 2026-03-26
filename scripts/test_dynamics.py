# scripts/test_dynamics.py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dynamics import QuadcopterDynamics

def run_test():
    dt = 0.01
    tau = 0.1 # Time constant for velocity tracking
    dynamics = QuadcopterDynamics(dt=dt, tau=tau)
    
    # Initial state
    dynamics.reset(np.array([0.0, 0.0, 10.0]))
    
    # Command: 5 m/s forward (x-axis)
    v_cmd = np.array([5.0, 0.0, 0.0, 0.0])
    
    # Simulation parameters
    duration = 2.0 # seconds
    steps = int(duration / dt)
    
    # Data storage
    time_history = []
    pos_history = []
    vel_history = []
    cmd_history = []
    
    print(f"Running dynamics test: Command = {v_cmd[0]} m/s for {duration}s")
    
    for i in range(steps):
        t = i * dt
        state = dynamics.step(v_cmd)
        
        time_history.append(t)
        pos_history.append(state[0:3].copy())
        vel_history.append(state[6:9].copy())
        cmd_history.append(v_cmd[0:3].copy())
        
    time_history = np.array(time_history)
    pos_history = np.array(pos_history)
    vel_history = np.array(vel_history)
    cmd_history = np.array(cmd_history)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    # 1. Velocity Tracking
    plt.subplot(3, 1, 1)
    plt.plot(time_history, vel_history[:, 0], label='Actual Vx', color='blue', linewidth=2)
    plt.plot(time_history, cmd_history[:, 0], '--', label='Commanded Vx', color='red')
    plt.title('Velocity Tracking Performance')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Position (Trajectory)
    plt.subplot(3, 1, 2)
    plt.plot(time_history, pos_history[:, 0], label='X Position', color='green')
    plt.title('Position Over Time')
    plt.ylabel('Position (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Tracking Error
    plt.subplot(3, 1, 3)
    error = cmd_history[:, 0] - vel_history[:, 0]
    plt.plot(time_history, error, label='Velocity Error', color='purple')
    plt.title('Tracking Error (Cmd - Actual)')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m/s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_dir = 'logs/tests'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'dynamics_test.png'))
    print(f"Test complete. Plot saved to {output_dir}/dynamics_test.png")
    
    # Analysis
    # 1. Delay (Time to reach 63.2% of command - 1 tau)
    target_63 = 0.632 * v_cmd[0]
    reach_time = time_history[np.where(vel_history[:, 0] >= target_63)[0][0]]
    
    # 2. Steady State Error
    ss_error = error[-1]
    
    # 3. Oscillation (Check for overshoot)
    max_vel = np.max(vel_history[:, 0])
    overshoot = max(0, (max_vel - v_cmd[0]) / v_cmd[0] * 100)
    
    print("\n--- Analysis ---")
    print(f"Rise Time (to 63.2%): {reach_time:.3f} s (Expected ~{tau}s)")
    print(f"Steady State Error: {ss_error:.5f} m/s")
    print(f"Overshoot: {overshoot:.2f} %")
    
    if overshoot < 0.1 and ss_error < 0.01:
        print("Result: Smooth, critically damped tracking (as expected for first-order dynamics).")
    else:
        print("Result: Non-ideal tracking detected.")

if __name__ == "__main__":
    run_test()
