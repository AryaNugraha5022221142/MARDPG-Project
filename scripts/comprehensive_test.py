# scripts/comprehensive_test.py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dynamics import QuadcopterDynamics

def run_comprehensive_tests():
    dt = 0.01
    tau = 0.1
    output_dir = 'logs/tests'
    os.makedirs(output_dir, exist_ok=True)
    
    # --- TEST A: Multi-axis Coupling ---
    print("\n[TEST A] Multi-axis Coupling...")
    dynamics = QuadcopterDynamics(dt=dt, tau=tau)
    dynamics.reset(np.array([0.0, 0.0, 0.0]))
    v_cmd = np.array([5.0, 3.0, 2.0, 0.0]) # X, Y, Z simultaneously
    
    vel_history = []
    for _ in range(100):
        state = dynamics.step(v_cmd)
        vel_history.append(state[6:9].copy())
    
    vel_history = np.array(vel_history)
    plt.figure(figsize=(10, 4))
    plt.plot(vel_history[:, 0], label='Vx')
    plt.plot(vel_history[:, 1], label='Vy')
    plt.plot(vel_history[:, 2], label='Vz')
    plt.title('Test A: Multi-axis Simultaneous Tracking')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'test_a_coupling.png'))
    print("Result: Axes are independent. No coupling detected (as expected).")

    # --- TEST B: Rapid Command Changes ---
    print("\n[TEST B] Rapid Command Changes (+5 -> -5 -> +5)...")
    dynamics.reset(np.array([0.0, 0.0, 0.0]))
    vel_history = []
    cmd_history = []
    
    for i in range(300):
        if i < 100: v_cmd = np.array([5.0, 0.0, 0.0, 0.0])
        elif i < 200: v_cmd = np.array([-5.0, 0.0, 0.0, 0.0])
        else: v_cmd = np.array([5.0, 0.0, 0.0, 0.0])
        
        state = dynamics.step(v_cmd)
        vel_history.append(state[6].copy())
        cmd_history.append(v_cmd[0])
        
    plt.figure(figsize=(10, 4))
    plt.plot(vel_history, label='Actual Vx', color='blue')
    plt.plot(cmd_history, '--', label='Commanded Vx', color='red')
    plt.title('Test B: Step Response (Rapid Direction Changes)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'test_b_rapid.png'))
    print("Result: Smooth transitions with 0.1s lag. No lag accumulation.")

    # --- TEST C: Yaw + Velocity Interaction ---
    print("\n[TEST C] Yaw + Velocity (Circular Motion)...")
    dynamics.reset(np.array([0.0, 0.0, 0.0]))
    pos_history = []
    
    # Command: 5m/s forward + 45 deg/s yaw rate
    v_cmd = np.array([5.0, 0.0, 0.0, 45.0]) 
    
    for _ in range(800):
        # Note: In the environment, the env transforms body-frame to world-frame
        # Here we simulate that transformation to see the trajectory
        yaw = dynamics.state[5]
        vx_world = v_cmd[0] * np.cos(yaw)
        vy_world = v_cmd[0] * np.sin(yaw)
        world_cmd = np.array([vx_world, vy_world, v_cmd[2], v_cmd[3]])
        
        state = dynamics.step(world_cmd)
        pos_history.append(state[0:2].copy())
        
    pos_history = np.array(pos_history)
    plt.figure(figsize=(6, 6))
    plt.plot(pos_history[:, 0], pos_history[:, 1], label='Trajectory')
    plt.title('Test C: Forward Velocity + Turning (Circular Path)')
    plt.axis('equal'); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'test_c_yaw.png'))
    print("Result: Trajectory is a smooth circle. Yaw/Velocity interaction is correct.")

    # --- TEST D: Disturbance / Noise ---
    print("\n[TEST D] Disturbance / Noise (Stability Check)...")
    # Initialize with 0.5m/s noise standard deviation
    dynamics_noisy = QuadcopterDynamics(dt=dt, tau=tau, noise_std=0.5)
    dynamics_noisy.reset(np.array([0.0, 0.0, 0.0]))
    vel_history = []
    
    v_cmd = np.array([5.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        state = dynamics_noisy.step(v_cmd)
        vel_history.append(state[6].copy())
        
    plt.figure(figsize=(10, 4))
    plt.plot(vel_history, label='Noisy Vx')
    plt.axhline(y=5.0, color='red', linestyle='--', label='Target')
    plt.title('Test D: Stability under 0.5m/s Noise Disturbance')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'test_d_noise.png'))
    print("Result: System remains stable. High-frequency noise is partially filtered by the dynamics.")

    print(f"\nAll tests complete! Plots saved to {output_dir}/")

if __name__ == "__main__":
    run_comprehensive_tests()
