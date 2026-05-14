# scripts/ursina_view_env.py
import sys
import os
import random
import numpy as np
import colorsys

# Try to import ursina
try:
    from ursina import *
except ImportError:
    print("Please install ursina first by running: pip install ursina")
    sys.exit(1)

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.quadcopter_kinematic_env import QuadcopterKinematicEnv

def get_height_color(z, max_z=40.0):
    """Returns a rainbow Colormap based on height to match the paper's MATLAB plots"""
    normalized_z = np.clip(z / max_z, 0.0, 1.0)
    # Hue goes from 0.7 (blue) to 0.0 (red)
    h = 0.7 - (normalized_z * 0.7)
    rgb = colorsys.hls_to_rgb(h, 0.5, 1.0)
    return color.rgb(rgb[0]*255, rgb[1]*255, rgb[2]*255)

class UrsinaVisualizer:
    def __init__(self):
        self.app = Ursina(title='MARDPG Environment Visualizer')
        
        self.config = {
            'arena_size': [100.0, 100.0, 40.0],
            'num_obstacles': 25,
            'action_bound': np.pi / 6.0,
            'dt': 0.05
        }
        
        self.scenes = ['pillars', 'cylinders', 'forest', 'rings']
        self.current_scene_idx = 0
        
        self.env = QuadcopterKinematicEnv(num_agents=3, config=self.config, render_mode=None)
        
        self.entities = []
        self.drone_entities = []
        self.goal_entities = []
        
        # Camera
        self.camera = EditorCamera(enabled=True)
        self.camera.position = (50, 60, -50)
        self.camera.look_at((50, 0, 50))
        
        # UI
        self.scene_names = {
            'pillars': 'Scene-I: Square Columns',
            'cylinders': 'Scene-II: Cylindrical Obstacles',
            'forest': 'Scene-III: Simulated Forest (Abstract blobs)',
            'rings': 'Scene-IV: Circular (Ring) Obstacles'
        }
        
        self.info_text = Text(text='', position=(-0.85, 0.45), scale=1.5, background=True)
        self.controls_text = Text(
            text='Press 1,2,3,4: Switch Scenes | Right-click + drag: Rotate camera | WASD: Move | ESC: Quit',
            position=(-0.85, -0.45), scale=1, background=True
        )
        
        self.setup_lighting()
        self.switch_scene(0)
        
    def setup_lighting(self):
        sun = DirectionalLight()
        sun.look_at(Vec3(1, -1, 1))
        AmbientLight(color=color.rgb(100, 100, 100))
        
    def clear_scene(self):
        for e in self.entities:
            destroy(e)
        for d in self.drone_entities:
            destroy(d)
        for g in self.goal_entities:
            destroy(g)
            
        self.entities.clear()
        self.drone_entities.clear()
        self.goal_entities.clear()
        
    def build_scene(self):
        self.clear_scene()
        
        scene_id = self.scenes[self.current_scene_idx]
        self.info_text.text = f'{self.scene_names[scene_id]}'
        
        # Ground (pale yellow like the MATLAB plots)
        ground = Entity(model='plane', scale=(100, 1, 100), position=(50, -0.1, 50), 
                        color=color.rgb(245, 245, 220), collider='box')
        self.entities.append(ground)
        
        # Render obstacles
        for obs in self.env.obstacles:
            pos = obs['pos']
            if obs['type'] == 'box':
                size = obs['size']
                is_cyl = obs.get('is_cylinder', False)
                mdl = 'cylinder' if is_cyl else 'cube'
                
                # Stack bands to achieve the rainbow gradient look from MATLAB plotting
                n_bands = max(1, int(size[2] / 2))
                band_h = size[2] / n_bands
                start_z = pos[2] - size[2]/2 + band_h/2
                
                for i in range(n_bands):
                    z = start_z + i * band_h
                    ent = Entity(
                        model=mdl,
                        position=(pos[0], z, pos[1]),
                        scale=(size[0], band_h, size[1]),
                        color=get_height_color(z, self.config['arena_size'][2])
                    )
                    self.entities.append(ent)
            
            elif obs['type'] == 'sphere':
                radius = obs['radius']
                ent = Entity(
                    model='sphere',
                    position=(pos[0], pos[2], pos[1]),
                    scale=radius*2,
                    color=get_height_color(pos[2] + radius, self.config['arena_size'][2])
                )
                self.entities.append(ent)
                
        # Create drones
        drone_colors = [color.red, color.green, color.blue, color.magenta]
        for i in range(self.env.num_agents):
            drone = Entity(model='sphere', scale=2.0, color=drone_colors[i % len(drone_colors)])
            self.drone_entities.append(drone)
            
            goal = Entity(model='cube', scale=2.0, color=color.black, alpha=0.5)
            self.goal_entities.append(goal)
            
    def update(self):
        # Step the environment with random actions
        actions = np.random.uniform(-self.config['action_bound'], self.config['action_bound'], size=(3, 2))
        next_obs, rewards, terminated, truncated, info = self.env.step(actions)
        
        # Update positions
        for i in range(self.env.num_agents):
            pos = self.env.agents[i].state[0:3]
            self.drone_entities[i].position = (pos[0], pos[2], pos[1])
            
            goal_pos = self.env.goals[i]
            self.goal_entities[i].position = (goal_pos[0], goal_pos[2], goal_pos[1])
            
        if terminated or truncated:
            self.env.reset()
            
    def input(self, key):
        if key == '1':
            self.switch_scene(0)
        elif key == '2':
            self.switch_scene(1)
        elif key == '3':
            self.switch_scene(2)
        elif key == '4':
            self.switch_scene(3)
            
    def switch_scene(self, idx):
        if self.current_scene_idx == idx and getattr(self, 'entities', []):
            return
        self.current_scene_idx = idx
        self.env.set_scene_type(self.scenes[idx])
        self.env.reset()
        self.build_scene()

if __name__ == '__main__':
    visualizer = UrsinaVisualizer()
    
    def update():
        visualizer.update()
        
    visualizer.app.run()
