# test_optimizer.py
import json
from optimizer import LayoutOptimizer

# Load input layout
with open('FineTuning/layouts/room-layout (2).json') as f:
    layout = json.load(f)

# Optimize
optimizer = LayoutOptimizer(layout)
optimized = optimizer.optimize(max_iterations=100)

# Save optimized layout
output_file = 'room-layout-optimized.json'
print(f"\n✓ Saved to {output_file}")