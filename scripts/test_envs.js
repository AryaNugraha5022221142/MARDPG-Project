const { execSync } = require('child_process');
try {
  console.log(execSync('python scripts/check_all_envs.py', { encoding: 'utf-8' }));
} catch (e) {
  console.log(e.stdout);
}
