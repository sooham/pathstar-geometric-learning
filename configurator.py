"""
Simple script to override default config values from command line.

Usage:
$ python train.py --batch_size=32 --learning_rate=1e-4
"""

import sys

# Parse command line arguments
for arg in sys.argv[1:]:
    if '=' not in arg:
        # Assume it's a boolean flag
        assert arg.startswith('--')
        key = arg[2:]
        if key in globals():
            globals()[key] = True
    else:
        assert arg.startswith('--')
        key, val = arg[2:].split('=')
        
        # Try to infer the type from the default value
        if key in globals():
            default_val = globals()[key]
            if isinstance(default_val, bool):
                globals()[key] = val.lower() in ['true', '1', 'yes']
            elif isinstance(default_val, int):
                globals()[key] = int(val)
            elif isinstance(default_val, float):
                globals()[key] = float(val)
            else:
                globals()[key] = val
        else:
            # If key doesn't exist, try to infer type from value
            try:
                globals()[key] = int(val)
            except ValueError:
                try:
                    globals()[key] = float(val)
                except ValueError:
                    if val.lower() in ['true', 'false']:
                        globals()[key] = val.lower() == 'true'
                    else:
                        globals()[key] = val

