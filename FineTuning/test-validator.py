# test_validator.py
import json
from validator import LayoutValidator

# Load your sample layout
with open('room-layout-optimized.json') as f:
    layout = json.load(f)

validator = LayoutValidator(layout)
violations = validator.validate()

print(f"Found {len(violations)} violations:")
for v in violations:
    print(f"  - {v}")