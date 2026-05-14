# scripts/ursina_view_env.py
import sys
import os
import random
import math
import numpy as np

try:
    from ursina import *
    from ursina.prefabs.first_person_controller import FirstPersonController
except ImportError:
    print("Please install ursina first: pip install ursina")
    sys.exit(1)

try:
    from noise import pnoise2
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False
    print("Note: 'noise' library not installed. Install with: pip install noise")
    print("Using simplified terrain generation.")

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.quadcopter_kinematic_env import QuadcopterKinematicEnv

class EnvironmentScene:
    """Base class for environment scenes"""
    
    def __init__(self):
        self.entities = []
        self.lights = []
    
    def clear(self):
        """Remove all entities from the scene"""
        for entity in self.entities:
            destroy(entity)
        for light in self.lights:
            destroy(light)
        self.entities.clear()
        self.lights.clear()
    
    def add_entity(self, entity):
        """Track entity for later cleanup"""
        self.entities.append(entity)
        return entity
    
    def add_light(self, light):
        """Track light for later cleanup"""
        self.lights.append(light)
        return light
    
    def setup_lighting(self):
        """Setup default lighting for the scene"""
        # Directional light (sun)
        sun = DirectionalLight(shadows=True)
        sun.look_at(Vec3(-1, -1, -1))
        self.add_light(sun)
        
        # Ambient light
        ambient = AmbientLight(color=color.rgb(50, 50, 50))
        self.add_light(ambient)
        
        return sun
    
    def create_ground(self, texture_name, scale=(100, 1, 100), color=color.white, repeat=10):
        """Create a ground plane with texture"""
        ground = Entity(
            model='plane',
            texture=texture_name,
            scale=scale,
            color=color,
            double_sided=False,
            collider='box',
            position=(0, -0.1, 0)
        )
        ground.texture_scale = (repeat, repeat)
        self.add_entity(ground)
        return ground
    
    def create_tree(self, x, z, trunk_color=color.rgb(101, 67, 33), 
                    foliage_color=color.rgb(34, 139, 34), scale=0.5):
        """Create a simple realistic tree"""
        trunk = Entity(
            model='cylinder',
            color=trunk_color,
            position=(x, 0, z),
            scale=(scale*0.3, scale, scale*0.3),
            collider='box'
        )
        # Foliage - multiple layers
        foliage1 = Entity(
            model='sphere',
            color=foliage_color,
            position=(x, scale*0.8, z),
            scale=(scale*0.8, scale*0.8, scale*0.8)
        )
        foliage2 = Entity(
            model='sphere',
            color=foliage_color,
            position=(x, scale*1.2, z),
            scale=(scale*0.6, scale*0.6, scale*0.6)
        )
        self.add_entity(trunk)
        self.add_entity(foliage1)
        self.add_entity(foliage2)
    
    def create_cactus(self, x, z, scale=0.4):
        """Create a saguaro cactus"""
        main_body = Entity(
            model='cylinder',
            color=color.rgb(34, 139, 34),
            position=(x, 0, z),
            scale=(scale*0.2, scale, scale*0.2),
            collider='box'
        )
        # Arms
        arm_left = Entity(
            model='cylinder',
            color=color.rgb(34, 139, 34),
            position=(x - scale*0.15, scale*0.6, z),
            scale=(scale*0.1, scale*0.3, scale*0.1),
            rotation=(0, 0, 45)
        )
        arm_right = Entity(
            model='cylinder',
            color=color.rgb(34, 139, 34),
            position=(x + scale*0.15, scale*0.6, z),
            scale=(scale*0.1, scale*0.3, scale*0.1),
            rotation=(0, 0, -45)
        )
        self.add_entity(main_body)
        self.add_entity(arm_left)
        self.add_entity(arm_right)
    
    def create_palm_tree(self, x, z, scale=0.5):
        """Create a palm tree"""
        trunk = Entity(
            model='cylinder',
            color=color.rgb(101, 67, 33),
            position=(x, 0, z),
            scale=(scale*0.15, scale, scale*0.15),
            rotation=(0, 0, 5)
        )
        # Fronds (using flattened cones)
        for angle in [0, 72, 144, 216, 288]:
            frond = Entity(
                model='cone',
                color=color.rgb(34, 139, 34),
                position=(x, scale*0.9, z),
                scale=(scale*0.5, scale*0.05, scale*0.1),
                rotation=(45, angle, 0)
            )
            self.add_entity(frond)
        self.add_entity(trunk)
    
    def create_rock(self, x, z, size=0.2):
        """Create a random rock"""
        rock = Entity(
            model='sphere',
            color=color.rgb(100, 100, 100),
            position=(x, size*0.3, z),
            scale=(size, size*0.6, size),
            collider='sphere'
        )
        self.add_entity(rock)

    def build(self):
        pass


class ForestScene(EnvironmentScene):
    """Scene-I: Lush Forest Environment"""
    def build(self):
        self.clear()
        self.setup_lighting()
        
        scene.fog_color = color.rgb(100, 120, 100)
        scene.fog_density = 0.01
        
        ground = self.create_ground('grass', scale=(120, 1, 120), repeat=15)
        ground.color = color.rgb(50, 80, 30)
        
        sky = Sky(texture='sky_default')
        self.add_entity(sky)
        
        for _ in range(150):
            x = random.uniform(-50, 50)
            z = random.uniform(-50, 50)
            if abs(x) < 15 and abs(z) < 15:
                continue
            scale = random.uniform(2, 6)
            self.create_tree(x, z, scale=scale)
        
        pond = Entity(
            model='cylinder',
            color=color.rgb(30, 100, 150),
            position=(0, -0.05, 0),
            scale=(15, 0.1, 15),
            alpha=0.8
        )
        self.add_entity(pond)


class DesertScene(EnvironmentScene):
    """Scene-II: Desert Oasis Environment"""
    def build(self):
        self.clear()
        sun = self.setup_lighting()
        sun.look_at(Vec3(-1, -0.5, -1))
        sun.color = color.rgb(255, 200, 150)
        
        scene.fog_color = color.rgb(200, 160, 100)
        scene.fog_density = 0.008
        
        ground = self.create_ground('sand', scale=(120, 1, 120), repeat=20)
        ground.color = color.rgb(210, 180, 140)
        
        sky = Sky(texture='sky_sunset')
        self.add_entity(sky)
        
        for _ in range(80):
            x = random.uniform(-55, 55)
            z = random.uniform(-55, 55)
            scale = random.uniform(2, 6)
            self.create_cactus(x, z, scale)
            
        oasis_water = Entity(
            model='cylinder',
            color=color.rgb(70, 150, 200),
            position=(0, -0.05, 0),
            scale=(15, 0.05, 15),
            alpha=0.9
        )
        self.add_entity(oasis_water)


class SnowyMountainsScene(EnvironmentScene):
    """Scene-III: Snowy Mountains Environment"""
    def build(self):
        self.clear()
        sun = self.setup_lighting()
        sun.look_at(Vec3(-1, -1.5, -0.5))
        sun.color = color.rgb(180, 200, 255)
        
        scene.fog_color = color.rgb(150, 170, 200)
        scene.fog_density = 0.015
        
        ground = self.create_ground('white', scale=(120, 1, 120), repeat=10)
        ground.color = color.rgb(220, 230, 240)
            
        sky = Sky(texture='sky_cloudy')
        self.add_entity(sky)
        
        for _ in range(100):
            x = random.uniform(-55, 55)
            z = random.uniform(-55, 55)
            scale = random.uniform(2, 6)
            trunk = Entity(
                model='cylinder',
                color=color.rgb(80, 60, 40),
                position=(x, 0, z),
                scale=(scale*0.2, scale, scale*0.2)
            )
            foliage = Entity(
                model='cone',
                color=color.rgb(200, 210, 220),
                position=(x, scale*0.8, z),
                scale=(scale*0.5, scale*0.6, scale*0.5)
            )
            self.add_entity(trunk)
            self.add_entity(foliage)


class BeachScene(EnvironmentScene):
    """Scene-IV: Tropical Beach Environment"""
    def build(self):
        self.clear()
        sun = self.setup_lighting()
        sun.look_at(Vec3(-1, -0.3, -1))
        sun.color = color.rgb(255, 240, 200)
        
        scene.fog_color = color.rgb(180, 200, 220)
        scene.fog_density = 0.005
        
        beach = self.create_ground('sand', scale=(120, 1, 120), repeat=15)
        beach.color = color.rgb(240, 220, 180)
        
        water = Entity(
            model='plane',
            texture='water',
            color=color.rgb(30, 140, 210),
            position=(0, -0.05, -30),
            scale=(120, 1, 80),
            alpha=0.85,
            double_sided=False
        )
        self.add_entity(water)
        
        sky = Sky(texture='sky_cloudy')
        self.add_entity(sky)
        
        for angle in range(-60, 61, 20):
            rad = angle * math.pi / 180
            x = math.cos(rad) * 25
            z = math.sin(rad) * 25 - 15
            self.create_palm_tree(x, z, scale=4.0)

class EnvironmentApp:
    def __init__(self):
        self.app = Ursina(borderless=False, title='Realistic MARDPG Environment Vis')
        
        self.camera = EditorCamera(enabled=True, rotation_speed=150, pan_speed=10)
        self.camera.position = (0, 20, -40)
        self.camera.look_at((0, 0, 0))
        
        self.scenes = {
            '1': ForestScene(),
            '2': DesertScene(),
            '3': SnowyMountainsScene(),
            '4': BeachScene()
        }
        self.current_scene = None
        
        self.scene_text = Text(text='', position=(-0.85, 0.45), scale=1.5, background=True)
        self.controls_text = Text(
            text='Press 1: Forest | 2: Desert | 3: Snowy Mountains | 4: Beach\n'
                 'Right-click + drag to rotate | WASD to move | ESC to exit',
            position=(-0.85, -0.45), scale=1, background=True
        )
        
        # MARDPG Environment
        self.config = {
            'arena_size': [100.0, 100.0, 40.0],
            'num_obstacles': 25,
            'action_bound': np.pi / 6.0,
            'dt': 0.05,
            'rangefinder_max_range': 30.0,
            'collision_distance': 0.8,
            'goal_distance': 2.0,
            'dynamic_ratio': 0.0
        }
        self.env = QuadcopterKinematicEnv(num_agents=3, config=self.config, render_mode=None)
        
        self.drone_entities = []
        self.goal_entities = []
        
        self.switch_scene('1')
        
    def switch_scene(self, scene_key):
        if self.current_scene == scene_key:
            return
            
        print(f"Switching to Scene-{scene_key}")
        
        if self.current_scene in self.scenes:
            self.scenes[self.current_scene].clear()
            
        self.current_scene = scene_key
        self.scenes[scene_key].build()
        
        scene_names = {'1': 'Lush Forest', '2': 'Desert Oasis', '3': 'Snowy Mountains', '4': 'Tropical Beach'}
        self.scene_text.text = f'Scene: {scene_names[scene_key]}'
        
        # Reset drones and recreate them
        for d in self.drone_entities:
            destroy(d)
        for g in self.goal_entities:
            destroy(g)
        self.drone_entities.clear()
        self.goal_entities.clear()
        
        self.env.reset()
        drone_colors = [color.red, color.green, color.blue, color.magenta]
        for i in range(self.env.num_agents):
            drone = Entity(model='sphere', scale=1.5, color=drone_colors[i % len(drone_colors)])
            self.drone_entities.append(drone)
            goal = Entity(model='cube', scale=1.5, color=color.black, alpha=0.5)
            self.goal_entities.append(goal)

    def update(self):
        if not self.drone_entities:
            return
            
        # Step MARDPG simulation
        actions = np.random.uniform(-self.config['action_bound'], self.config['action_bound'], size=(3, 2))
        next_obs, rewards, terminated, truncated, info = self.env.step(actions)
        
        # Center coordinates around 0,0 since arena is 0-100 and Ursina is centered.
        # We'll map (0->100) to (-50->50)
        offset = 50.0
        
        for i in range(self.env.num_agents):
            pos = self.env.agents[i].state[0:3]
            # y in Ursina is Up, z is Depth. 
            self.drone_entities[i].position = (pos[0]-offset, pos[2], pos[1]-offset)
            
            goal_pos = self.env.goals[i]
            self.goal_entities[i].position = (goal_pos[0]-offset, goal_pos[2], goal_pos[1]-offset)
            
        if terminated or truncated:
            self.env.reset()

    def run(self):
        # We need to hook up our update function to the global Ursina update
        # We can do this simply by defining a global update function
        global update
        def update():
            self.update()
            
        global input
        def input(key):
            if key in ['1', '2', '3', '4']:
                self.switch_scene(key)
        
        self.app.run()

if __name__ == '__main__':
    app = EnvironmentApp()
    app.run()

