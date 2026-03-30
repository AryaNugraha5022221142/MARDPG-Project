import React, { useState } from 'react';
import QuadcopterCanvas from './components/QuadcopterVisualizer';
import { 
  Activity, 
  Cpu, 
  Layers, 
  Play, 
  Pause, 
  RotateCcw, 
  Settings, 
  Target, 
  Wind,
  Zap,
  Map as MapIcon,
  TreePine,
  Building2,
  AlertTriangle
} from 'lucide-react';

export default function App() {
  const [scenario, setScenario] = useState('city');
  const [numAgents, setNumAgents] = useState(3);
  const [isSimulating, setIsSimulating] = useState(false);
  const [key, setKey] = useState(0); // For resetting simulation

  const resetSimulation = () => {
    setIsSimulating(false);
    setKey(prev => prev + 1);
  };

  const scenarios = [
    { id: 'city', name: 'Urban Canyon', icon: <Building2 className="w-4 h-4" />, desc: 'High-density buildings, narrow passages.' },
    { id: 'forest', name: 'Dense Forest', icon: <TreePine className="w-4 h-4" />, desc: 'Complex organic obstacles, low visibility.' },
    { id: 'dynamic_chaos', name: 'Dynamic Chaos', icon: <Zap className="w-4 h-4" />, desc: 'Moving obstacles, high collision risk.' },
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#e0e0e0] font-mono selection:bg-[#f27d26] selection:text-black">
      {/* Header / Top Bar */}
      <header className="border-b border-white/10 p-4 flex items-center justify-between bg-black/50 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-[#f27d26] rounded-sm flex items-center justify-center text-black">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tighter uppercase italic">MARDPG-Project</h1>
            <p className="text-[10px] text-white/40 uppercase tracking-widest">Multi-Agent Deep Reinforcement Learning</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="flex flex-col items-end">
            <span className="text-[10px] text-white/40 uppercase tracking-widest">System Status</span>
            <span className="text-xs text-[#2ecc71] flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#2ecc71] animate-pulse" />
              OPERATIONAL
            </span>
          </div>
          <div className="w-px h-8 bg-white/10" />
          <button className="p-2 hover:bg-white/5 rounded-full transition-colors">
            <Settings className="w-5 h-5 text-white/60" />
          </button>
        </div>
      </header>

      <main className="p-6 grid grid-cols-12 gap-6 max-w-[1600px] mx-auto">
        {/* Sidebar Controls */}
        <div className="col-span-12 lg:col-span-3 space-y-6">
          {/* Simulation Controls */}
          <section className="bg-white/5 border border-white/10 rounded-lg p-5 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-[#f27d26]" />
              <h2 className="text-xs font-bold uppercase tracking-widest text-white/80">Simulation Control</h2>
            </div>
            
            <div className="grid grid-cols-2 gap-3">
              <button 
                onClick={() => setIsSimulating(!isSimulating)}
                className={`flex items-center justify-center gap-2 py-3 rounded text-xs font-bold uppercase transition-all ${
                  isSimulating 
                    ? 'bg-white/10 text-white border border-white/20 hover:bg-white/20' 
                    : 'bg-[#f27d26] text-black hover:bg-[#ff8c3a]'
                }`}
              >
                {isSimulating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isSimulating ? 'Pause' : 'Start'}
              </button>
              <button 
                onClick={resetSimulation}
                className="flex items-center justify-center gap-2 py-3 rounded text-xs font-bold uppercase bg-white/5 text-white border border-white/10 hover:bg-white/10"
              >
                <RotateCcw className="w-4 h-4" />
                Reset
              </button>
            </div>

            <div className="pt-4 border-t border-white/5 space-y-4">
              <div>
                <label className="text-[10px] text-white/40 uppercase tracking-widest block mb-2">Agent Count: {numAgents}</label>
                <input 
                  type="range" 
                  min="1" 
                  max="6" 
                  value={numAgents} 
                  onChange={(e) => setNumAgents(parseInt(e.target.value))}
                  className="w-full accent-[#f27d26] bg-white/10 h-1 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            </div>
          </section>

          {/* Scenario Selection */}
          <section className="bg-white/5 border border-white/10 rounded-lg p-5 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <MapIcon className="w-4 h-4 text-[#f27d26]" />
              <h2 className="text-xs font-bold uppercase tracking-widest text-white/80">Environment Scenarios</h2>
            </div>
            
            <div className="space-y-2">
              {scenarios.map((s) => (
                <button
                  key={s.id}
                  onClick={() => { setScenario(s.id); resetSimulation(); }}
                  className={`w-full text-left p-3 rounded border transition-all flex items-start gap-3 ${
                    scenario === s.id 
                      ? 'bg-[#f27d26]/10 border-[#f27d26] text-white' 
                      : 'bg-white/5 border-transparent text-white/60 hover:bg-white/10'
                  }`}
                >
                  <div className={`p-2 rounded ${scenario === s.id ? 'bg-[#f27d26] text-black' : 'bg-white/10 text-white/60'}`}>
                    {s.icon}
                  </div>
                  <div>
                    <div className="text-xs font-bold uppercase">{s.name}</div>
                    <div className="text-[10px] opacity-60 leading-tight mt-1">{s.desc}</div>
                  </div>
                </button>
              ))}
            </div>
          </section>

          {/* Telemetry Mockup */}
          <section className="bg-white/5 border border-white/10 rounded-lg p-5">
            <div className="flex items-center gap-2 mb-4">
              <Wind className="w-4 h-4 text-[#f27d26]" />
              <h2 className="text-xs font-bold uppercase tracking-widest text-white/80">Telemetry Data</h2>
            </div>
            <div className="space-y-3">
              {[
                { label: 'Avg Velocity', value: isSimulating ? '4.2 m/s' : '0.0 m/s', color: 'text-white' },
                { label: 'Collision Risk', value: isSimulating ? 'LOW' : 'NONE', color: 'text-[#2ecc71]' },
                { label: 'Battery Level', value: '98%', color: 'text-white' },
                { label: 'GPS Signal', value: 'LOCKED', color: 'text-[#2ecc71]' },
              ].map((item, i) => (
                <div key={i} className="flex justify-between items-center border-b border-white/5 pb-2">
                  <span className="text-[10px] text-white/40 uppercase">{item.label}</span>
                  <span className={`text-xs font-bold ${item.color}`}>{item.value}</span>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Main Viewport */}
        <div className="col-span-12 lg:col-span-9 flex flex-col gap-6">
          <div className="relative aspect-video lg:aspect-auto lg:h-[700px]">
            <QuadcopterCanvas key={key} scenario={scenario} numAgents={numAgents} isSimulating={isSimulating} />
            
            {/* Overlay UI */}
            <div className="absolute top-4 left-4 flex flex-col gap-2 pointer-events-none">
              <div className="bg-black/80 backdrop-blur-md border border-white/10 px-3 py-1.5 rounded text-[10px] font-bold text-[#f27d26] uppercase tracking-widest flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-[#f27d26] animate-pulse" />
                Live Feed: Scenario_{scenario.toUpperCase()}
              </div>
              <div className="bg-black/80 backdrop-blur-md border border-white/10 px-3 py-1.5 rounded text-[10px] font-bold text-white/60 uppercase tracking-widest">
                Camera: Orbit_Free_Cam
              </div>
            </div>

            <div className="absolute bottom-4 right-4 bg-black/80 backdrop-blur-md border border-white/10 p-4 rounded-lg pointer-events-none">
              <div className="flex items-center gap-4">
                <div className="text-center">
                  <div className="text-[10px] text-white/40 uppercase mb-1">X-Pos</div>
                  <div className="text-lg font-bold text-white tabular-nums">100.0</div>
                </div>
                <div className="w-px h-8 bg-white/10" />
                <div className="text-center">
                  <div className="text-[10px] text-white/40 uppercase mb-1">Y-Pos</div>
                  <div className="text-lg font-bold text-white tabular-nums">100.0</div>
                </div>
                <div className="w-px h-8 bg-white/10" />
                <div className="text-center">
                  <div className="text-[10px] text-white/40 uppercase mb-1">Z-Pos</div>
                  <div className="text-lg font-bold text-white tabular-nums">40.0</div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Data Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white/5 border border-white/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <Layers className="w-4 h-4 text-[#f27d26]" />
                <h3 className="text-[10px] font-bold uppercase tracking-widest">Environment Info</h3>
              </div>
              <p className="text-[11px] text-white/60 leading-relaxed">
                Arena Size: 100m x 100m x 40m<br />
                Obstacle Type: {scenario === 'city' ? 'AABB (Buildings)' : 'Spheres/Cylinders'}<br />
                Gravity: 9.81 m/s²<br />
                Air Density: 1.225 kg/m³
              </p>
            </div>
            
            <div className="bg-white/5 border border-white/10 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <Target className="w-4 h-4 text-[#f27d26]" />
                <h3 className="text-[10px] font-bold uppercase tracking-widest">Mission Parameters</h3>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-[10px] text-white/40 uppercase">Success Rate</span>
                  <div className="w-24 h-1 bg-white/10 rounded-full overflow-hidden">
                    <div className="w-[85%] h-full bg-[#2ecc71]" />
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[10px] text-white/40 uppercase">Avg. Reward</span>
                  <span className="text-[10px] font-bold text-[#f1c40f]">+124.5</span>
                </div>
              </div>
            </div>

            <div className="bg-white/5 border border-white/10 rounded-lg p-4 flex items-center gap-4">
              <div className="w-12 h-12 rounded-full border-2 border-dashed border-[#f27d26]/40 flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-[#f27d26]" />
              </div>
              <div>
                <h3 className="text-[10px] font-bold uppercase tracking-widest">Safety Protocol</h3>
                <p className="text-[10px] text-white/40 mt-1">Collision avoidance active. Minimum separation: 2.0m.</p>
              </div>
            </div>
          </div>
        </div>
      </main>
      
      {/* Footer */}
      <footer className="mt-12 border-t border-white/10 p-6 text-center">
        <p className="text-[10px] text-white/20 uppercase tracking-[0.3em]">
          &copy; 2026 MARDPG Research Lab // Autonomous Systems Division
        </p>
      </footer>
    </div>
  );
}
