import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import main

def _translate_legacy_args():
    translated = []
    for arg in sys.argv:
        if arg == '--num_agents':
            translated.append('--num-agents')
        else:
            translated.append(arg)
    sys.argv = translated

if __name__ == '__main__':
    _translate_legacy_args()
    main()
