# test_optimizer.py
import json
from optimizer import LayoutOptimizer

# Load input layout
with open('FineTuning/layouts/room-layout (43).json') as f:
    layout = json.load(f)

# Optimize
optimizer = LayoutOptimizer(layout)
optimized = optimizer.optimize(max_iterations=100)

# Save optimized layout
output_file = 'room-layout-optimized.json'
with open(output_file, 'w') as f:
    json.dump(optimized, f, indent=2)

print("\n=== OPTIMIZED LAYOUT ===")
print(json.dumps(optimized, indent=2))
print(f"\n✓ Saved to {output_file}")