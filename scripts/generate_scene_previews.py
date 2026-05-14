import os
import sys
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.quadcopter_kinematic_env import QuadcopterKinematicEnv

def main():
    os.makedirs('static', exist_ok=True)
    scenes = ['pillars', 'cylinders', 'forest', 'rings']
    
    config = {
        'arena_size': [100.0, 100.0, 40.0],
        'num_obstacles': 25,
        'rangefinder_max_range': 30.0,
        'collision_distance': 0.8,
        'goal_distance': 2.0,
        'dt': 0.01,
        'dynamic_ratio': 0.0
    }
    
    env = QuadcopterKinematicEnv(num_agents=3, config=config, render_mode='human')
    
    for scene in scenes:
        print(f"Generating preview for {scene}...")
        env.set_scene_type(scene)
        env.reset()
        
        # Set save path so it doesn't try to open an interactive window
        env.save_path = f"static/{scene}.png"
        env.render()
        
        # We need to manually clear the figure between renders or create a new one
        if env.fig:
            plt.close(env.fig)
            env.fig = None
            env.ax = None
            
    # Generate an index.html file to view them easily
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>MARDPG Environment Scenes</title>
    <style>
        body { font-family: system-ui, sans-serif; text-align: center; background: #1a1a1a; color: #fff; padding: 20px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1200px; margin: 0 auto; }
        .card { background: #2a2a2a; padding: 10px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        img { width: 100%; border-radius: 4px; }
        h1 { margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>Multi-Agent Navigation Environments</h1>
    <div class="grid">
        <div class="card">
            <h2>Scene I - Square Columns (Pillars)</h2>
            <img src="pillars.png" alt="Pillars">
        </div>
        <div class="card">
            <h2>Scene II - Cylinders</h2>
            <img src="cylinders.png" alt="Cylinders">
        </div>
        <div class="card">
            <h2>Scene III - Forest Obstacles</h2>
            <img src="forest.png" alt="Forest">
        </div>
        <div class="card">
            <h2>Scene IV - Circular Obstacles (Rings)</h2>
            <img src="rings.png" alt="Rings">
        </div>
    </div>
</body>
</html>'''
    
    with open('static/index.html', 'w') as f:
        f.write(html_content)
        
    print("Done! You can view these images by serving the 'static' directory.")

if __name__ == '__main__':
    main()
