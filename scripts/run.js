const { execSync } = require('child_process');
try {
  const result = execSync('python3 -m pip install pyyaml setuptools numpy; python3 debug_col.py', { encoding: 'utf-8' });
  console.log(result);
} catch (e) {
  console.error(e.stdout);
  console.error(e.stderr);
}
