import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid, Stars, Text, Float } from '@react-three/drei';
import * as THREE from 'three';

// Types
interface Obstacle {
  type: 'sphere' | 'box';
  pos: [number, number, number];
  size?: [number, number, number];
  radius?: number;
  color: string;
  opacity?: number;
}

interface Drone {
  id: number;
  pos: [number, number, number];
  yaw: number;
  goal: [number, number, number];
  color: string;
}

// Components
const DroneModel = ({ pos, yaw, color }: { pos: [number, number, number], yaw: number, color: string }) => {
  const groupRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.position.set(...pos);
      groupRef.current.rotation.z = yaw;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Body */}
      <mesh>
        <boxGeometry args={[0.4, 0.4, 0.1]} />
        <meshStandardMaterial color={color} />
      </mesh>
      {/* Arms */}
      <mesh rotation={[0, 0, Math.PI / 4]}>
        <boxGeometry args={[1.2, 0.1, 0.05]} />
        <meshStandardMaterial color="#333" />
      </mesh>
      <mesh rotation={[0, 0, -Math.PI / 4]}>
        <boxGeometry args={[1.2, 0.1, 0.05]} />
        <meshStandardMaterial color="#333" />
      </mesh>
      {/* Rotors */}
      {[-0.4, 0.4].map((x) => 
        [-0.4, 0.4].map((y) => (
          <mesh key={`${x}-${y}`} position={[x, y, 0.1]}>
            <cylinderGeometry args={[0.2, 0.2, 0.02, 16]} />
            <meshStandardMaterial color="#666" transparent opacity={0.6} />
          </mesh>
        ))
      )}
      {/* Direction indicator */}
      <mesh position={[0.3, 0, 0.05]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.1, 0.3, 16]} />
        <meshStandardMaterial color="white" />
      </mesh>
    </group>
  );
};

const GoalMarker = ({ pos, color }: { pos: [number, number, number], color: string }) => {
  return (
    <group position={pos}>
      <mesh>
        <sphereGeometry args={[0.5, 16, 16]} />
        <meshStandardMaterial color={color} transparent opacity={0.3} />
      </mesh>
      <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.8, 1.0, 32]} />
          <meshStandardMaterial color={color} side={THREE.DoubleSide} />
        </mesh>
      </Float>
      <Text
        position={[0, 0, 1.5]}
        fontSize={0.8}
        color={color}
        anchorX="center"
        anchorY="middle"
      >
        GOAL
      </Text>
    </group>
  );
};

const ObstacleModel = ({ obs }: { obs: Obstacle }) => {
  if (obs.type === 'sphere') {
    return (
      <mesh position={obs.pos}>
        <sphereGeometry args={[obs.radius || 1, 32, 32]} />
        <meshStandardMaterial 
          color={obs.color} 
          transparent 
          opacity={obs.opacity || 0.8} 
          roughness={0.3}
          metalness={0.2}
        />
      </mesh>
    );
  } else {
    return (
      <mesh position={obs.pos}>
        <boxGeometry args={obs.size || [1, 1, 1]} />
        <meshStandardMaterial 
          color={obs.color} 
          transparent 
          opacity={obs.opacity || 0.8}
          roughness={0.7}
          metalness={0.1}
        />
      </mesh>
    );
  }
};

const Arena = ({ size }: { size: [number, number, number] }) => {
  return (
    <group>
      {/* Floor */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[size[0] / 2, size[1] / 2, 0]}>
        <planeGeometry args={[size[0], size[1]]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>
      
      {/* Grid */}
      <Grid 
        position={[size[0] / 2, 0.01, size[1] / 2]} 
        args={[size[0], size[1]]} 
        cellColor="#444" 
        sectionColor="#666" 
        fadeDistance={150}
      />

      {/* Boundary Lines */}
      <lineSegments position={[size[0] / 2, size[1] / 2, size[2] / 2]}>
        <edgesGeometry args={[new THREE.BoxGeometry(size[0], size[1], size[2])]} />
        <lineBasicMaterial color="#333" />
      </lineSegments>
    </group>
  );
};

export const QuadcopterVisualizer = ({ 
  scenario = 'city', 
  numAgents = 3,
  isSimulating = false 
}: { 
  scenario?: string, 
  numAgents?: number,
  isSimulating?: boolean 
}) => {
  const arenaSize: [number, number, number] = [100, 100, 40];
  
  // Generate obstacles based on scenario
  const obstacles = useMemo(() => {
    const obs: Obstacle[] = [];
    const buildingColors = ['#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7'];
    
    if (scenario === 'city') {
      const spacing = 15.0;
      for (let x = 15.0; x < arenaSize[0] - 10.0; x += spacing) {
        for (let y = 15.0; y < arenaSize[1] - 10.0; y += spacing) {
          if (Math.random() < 0.7) {
            const w = 6 + Math.random() * 4;
            const d = 6 + Math.random() * 4;
            const h = 15 + Math.random() * 20;
            obs.push({
              type: 'box',
              pos: [x, y, h / 2],
              size: [w, d, h],
              color: buildingColors[Math.floor(Math.random() * buildingColors.length)]
            });
          }
        }
      }
    } else if (scenario === 'forest') {
      for (let i = 0; i < 60; i++) {
        const x = 5 + Math.random() * (arenaSize[0] - 10);
        const y = 5 + Math.random() * (arenaSize[1] - 10);
        const h = 8 + Math.random() * 10;
        // Trunk
        obs.push({
          type: 'box',
          pos: [x, y, h / 2],
          size: [0.8, 0.8, h],
          color: '#5d4037'
        });
        // Leaves
        obs.push({
          type: 'sphere',
          pos: [x, y, h],
          radius: 2 + Math.random() * 2,
          color: '#2e7d32',
          opacity: 0.9
        });
      }
    } else if (scenario === 'dynamic_chaos') {
      for (let i = 0; i < 20; i++) {
        obs.push({
          type: 'sphere',
          pos: [
            10 + Math.random() * 80,
            10 + Math.random() * 80,
            5 + Math.random() * 30
          ],
          radius: 2 + Math.random() * 3,
          color: '#e67e22'
        });
      }
    }
    
    return obs;
  }, [scenario]);

  // Initial drone states
  const [drones, setDrones] = useState<Drone[]>(() => {
    const colors = ['#e74c3c', '#2ecc71', '#3498db', '#f1c40f', '#9b59b6', '#1abc9c'];
    return Array.from({ length: numAgents }).map((_, i) => ({
      id: i,
      pos: [5 + i * 5, 5, 10],
      yaw: 0,
      goal: [arenaSize[0] - 10 - i * 5, arenaSize[1] - 10, 20],
      color: colors[i % colors.length]
    }));
  });

  // Simulation loop
  useFrame((state, delta) => {
    if (!isSimulating) return;

    setDrones(prev => prev.map(drone => {
      const target = drone.goal;
      const dx = target[0] - drone.pos[0];
      const dy = target[1] - drone.pos[1];
      const dz = target[2] - drone.pos[2];
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      
      if (dist < 1) return drone; // Reached goal

      const speed = 5 * delta;
      const vx = (dx / dist) * speed;
      const vy = (dy / dist) * speed;
      const vz = (dz / dist) * speed;

      // Simple collision avoidance with obstacles (very basic)
      let avoidX = 0, avoidY = 0, avoidZ = 0;
      obstacles.forEach(obs => {
        const ox = obs.pos[0] - drone.pos[0];
        const oy = obs.pos[1] - drone.pos[1];
        const oz = obs.pos[2] - drone.pos[2];
        const d = Math.sqrt(ox*ox + oy*oy + oz*oz);
        const minSafe = (obs.type === 'sphere' ? obs.radius || 1 : 5) + 2;
        
        if (d < minSafe) {
          avoidX -= (ox / d) * (minSafe - d) * 0.5;
          avoidY -= (oy / d) * (minSafe - d) * 0.5;
          avoidZ -= (oz / d) * (minSafe - d) * 0.5;
        }
      });

      return {
        ...drone,
        pos: [
          drone.pos[0] + vx + avoidX,
          drone.pos[1] + vy + avoidY,
          drone.pos[2] + vz + avoidZ
        ],
        yaw: Math.atan2(vy, vx)
      };
    }));
  });

  return (
    <group>
      <PerspectiveCamera makeDefault position={[120, 120, 80]} fov={50} />
      <OrbitControls target={[arenaSize[0] / 2, arenaSize[1] / 2, 10]} />
      
      <ambientLight intensity={0.5} />
      <pointLight position={[50, 50, 100]} intensity={1} />
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
      
      <Arena size={arenaSize} />
      
      {obstacles.map((obs, i) => (
        <ObstacleModel key={i} obs={obs} />
      ))}
      
      {drones.map(drone => (
        <React.Fragment key={drone.id}>
          <DroneModel pos={drone.pos} yaw={drone.yaw} color={drone.color} />
          <GoalMarker pos={drone.goal} color={drone.color} />
        </React.Fragment>
      ))}
    </group>
  );
};

export default function QuadcopterCanvas({ scenario, numAgents, isSimulating }: any) {
  return (
    <div className="w-full h-full bg-black rounded-xl overflow-hidden border border-white/10 shadow-2xl">
      <Canvas shadows>
        <QuadcopterVisualizer scenario={scenario} numAgents={numAgents} isSimulating={isSimulating} />
      </Canvas>
    </div>
  );
}
