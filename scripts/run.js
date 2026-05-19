const { execSync } = require('child_process');
try {
  const result = execSync('python3 scripts/debug_eval.py', { encoding: 'utf-8' });
  console.log(result);
} catch (e) {
  console.error(e.stdout);
  console.error(e.stderr);
}
