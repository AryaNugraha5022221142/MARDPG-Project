import argparse
import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.base_env import EnvironmentConfig, DifficultyLevel, ObstacleType
from envs.benchmark_suite import BenchmarkSuite

class MatplotlibVisualizer:
    def __init__(self, env):
        self.env = env
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()
        self.agent_scatter = None
        self.goal_scatter = None
        self.obstacle_collections = []
        self._init_plot()
        
    def _init_plot(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.env.config.map_width)
        self.ax.set_ylim(0, self.env.config.map_depth)
        self.ax.set_zlim(0, self.env.config.map_height)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Plot obstacles
        for obs in self.env.obstacles:
            color = obs.color
            if obs.obstacle_type == ObstacleType.CUBOID:
                self._plot_cuboid(obs.position, obs.dimensions, obs.rotation_deg, color)
            elif obs.obstacle_type in (ObstacleType.CYLINDER, ObstacleType.DYNAMIC):
                self._plot_cylinder(obs.position, obs.dimensions, color)
            elif obs.obstacle_type == ObstacleType.SPHERE:
                self._plot_sphere(obs.position, obs.dimensions, color)
                
        # Scatters for agents and goals
        self.agent_scatter = self.ax.scatter([], [], [], c='b', marker='^', s=100, label='Agents')
        self.goal_scatter = self.ax.scatter([], [], [], c='r', marker='x', s=100, label='Goals')
        self.ax.legend()
        
    def _plot_cuboid(self, pos, dim, rot, color):
        dx, dy, dz = dim
        base_corners = np.array([
            [-dx/2, -dy/2, 0],
            [dx/2, -dy/2, 0],
            [dx/2, dy/2, 0],
            [-dx/2, dy/2, 0]
        ])
        
        # Rotate
        rad = np.radians(rot)
        c, s = np.cos(rad), np.sin(rad)
        rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners = np.dot(base_corners, rot_mat.T)
        
        # Translate
        corners[:, 0] += pos[0]
        corners[:, 1] += pos[1]
        
        # Add Z
        z_base = pos[2]
        bot = np.c_[corners[:,:2], np.full(4, z_base)]
        top = np.c_[corners[:,:2], np.full(4, z_base + dz)]
        
        faces = [
            bot, top,
            [bot[0], bot[1], top[1], top[0]],
            [bot[1], bot[2], top[2], top[1]],
            [bot[2], bot[3], top[3], top[2]],
            [bot[3], bot[0], top[0], top[3]]
        ]
        
        poly3d = Poly3DCollection(faces, alpha=0.6, facecolors=color, linewidths=0.5, edgecolors='k')
        self.ax.add_collection3d(poly3d)
        
    def _plot_cylinder(self, pos, dim, color):
        r, _, h = dim
        z_base = pos[2]
        z = np.linspace(z_base, z_base + h, 2)
        theta = np.linspace(0, 2*np.pi, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = r * np.cos(theta_grid) + pos[0]
        y_grid = r * np.sin(theta_grid) + pos[1]
        self.ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.6)
        
    def _plot_sphere(self, pos, dim, color):
        rx, ry, rz = dim[0]/2, dim[1]/2, dim[2]/2
        u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
        x = rx * np.cos(u) * np.sin(v) + pos[0]
        y = ry * np.sin(u) * np.sin(v) + pos[1]
        z = rz * np.cos(v) + (pos[2] + rz)
        self.ax.plot_surface(x, y, z, color=color, alpha=0.6)
        
    def render(self):
        agents = np.array(self.env.agents)
        goals = np.array(self.env.goals)
        
        if len(agents) > 0:
            self.agent_scatter._offsets3d = (agents[:,0], agents[:,1], agents[:,2])
        if len(goals) > 0:
            self.goal_scatter._offsets3d = (goals[:,0], goals[:,1], goals[:,2])
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main():
    parser = argparse.ArgumentParser(description="View the 3D environments with random agent actions.")
    parser.add_argument('--scene', type=str, default='urban', choices=['urban', 'forest', 'terrain', 'structured', 'mixed', 'dynamic'], help="Type of scene to load.")
    parser.add_argument('--difficulty', type=str, default='EASY', choices=['TRIVIAL', 'EASY', 'MEDIUM', 'HARD', 'EXTREME'], help="Difficulty level.")
    parser.add_argument('--steps', type=int, default=1000, help="Number of simulation steps to run.")
    parser.add_argument('--dt', type=float, default=0.05, help="Time step delta for the visualizer.")
    args = parser.parse_args()

    difficulty = DifficultyLevel[args.difficulty]
    cfg = EnvironmentConfig(
        map_width=50.0,
        map_depth=50.0,
        map_height=20.0,
        difficulty=difficulty,
        goal_mode="opposite",
        uav_spawn_mode="corners",
        n_agents=4
    )
    
    env = BenchmarkSuite.make(args.scene, cfg)
    env.reset()
    
    print(f"Viewing scene: {args.scene} at {args.difficulty} difficulty")
    print(f"Number of obstacles: {len(env.obstacles)}")
    print("Close the matplotlib window or press Ctrl+C in the terminal to exit.")

    visualizer = MatplotlibVisualizer(env)
    
    try:
        max_speed = env.config.max_speed
        for _ in range(args.steps):
            # Generate random steering commands
            actions = np.random.uniform(-max_speed, max_speed, size=(cfg.n_agents, 3))
            
            # Step the environment
            env.step(actions, dt=args.dt)
            
            # Render the environment
            visualizer.render()
            
            # Reset logic if we want continuous visualization
            # Normally we'd reset when done, but for just viewing let them drift
            
            time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("Visualization stopped by user.")
        
    print("Done. Keeping window open.")
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
