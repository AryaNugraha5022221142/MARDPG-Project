const fs = require('fs');

let lines = fs.readFileSync('scripts/evaluate.py', 'utf8').split('\n');

// The `for ep in range(args.episodes):` is at line 103 (0-indexed).
// It has 12 spaces of indentation.
// The lines from 112 (`steps = 0`) to 191 (`plt.close(fig_traj)`) should all have +4 spaces because they are inside the `for ep` loop.
// The lines from 193 to 216 should be at 8 spaces (inside `for level in levels`).

for (let i = 112; i <= 191; i++) {
    lines[i] = '    ' + lines[i];
}
for (let i = 193; i <= 215; i++) {
    // These were indented at 8 but maybe they're at whatever. Let's force them to be 8 spaces.
    lines[i] = '        ' + lines[i].trimStart();
}

fs.writeFileSync('scripts/evaluate.py', lines.join('\n'));
console.log("Done");
