const { execSync } = require('child_process');
try {
  const result = execSync('python3 -m pip install pyyaml setuptools numpy; python3 scripts/evaluate.py --agent mardpg_baseline --checkpoint checkpoints/mardpg_baseline_final.pt --scenes urban --episodes 1', { encoding: 'utf-8' });
  console.log(result);
} catch (e) {
  console.error(e.stdout);
  console.error(e.stderr);
}
