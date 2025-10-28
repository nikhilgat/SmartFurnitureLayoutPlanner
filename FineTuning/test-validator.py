"""
Simple Validator Test - Just run it on your layout files
Usage: python simple_test.py your-layout.json
"""

import json
import sys

try:
    from validator import LayoutValidator
except ImportError:
    print("Error: Cannot import validator")
    print("Make sure validator.py is in the same directory")
    print("Install shapely: pip install shapely")
    sys.exit(1)


def test_layout(filename):
    """Test a layout file and print violations"""
    
    print("="*70)
    print(f" TESTING: {filename}")
    print("="*70)
    
    try:
        with open(filename, 'r') as f:
            layout = json.load(f)
        print(f"Loaded layout: {filename}")
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return
    except json.JSONDecodeError:
        print(f"Invalid JSON in: {filename}")
        return

    print(f"\nRoom: {layout['room']['width']}x{layout['room']['height']} cm")
    print(f"Furniture: {len(layout.get('furniture', []))} items")
    print(f"Openings: {len(layout.get('openings', []))} items")


    print("\nRunning validation...")
    validator = LayoutValidator(layout)
    violations = validator.validate()

    print("\n" + "=" * 70)
    print(f" RESULTS: {len(violations)} VIOLATIONS")
    print("=" * 70)

    if not violations:
        print("\nPerfect! No violations found.\n")
        return

    groups = {
        "Overlaps": [],
        "Clearances": [],
        "Bed Clearances": [],
        "Turning Space": [],
        "Door": [],
        "Emergency Path": [],
        "Windows": [],
        "Other": []
    }

    for v in violations:
        v_lower = v.lower()
        matched = False

        if "overlap" in v_lower:
            groups["Overlaps"].append(v)
            matched = True
        if "bed" in v_lower and "clearance" in v_lower:
            groups["Bed Clearances"].append(v)
            matched = True
        if "clearance" in v_lower and not matched:
            groups["Clearances"].append(v)
            matched = True
        if "turning" in v_lower:
            groups["Turning Space"].append(v)
            matched = True
        if "door" in v_lower:
            groups["Door"].append(v)
            matched = True
        if "emergency" in v_lower or "path" in v_lower:
            groups["Emergency Path"].append(v)
            matched = True
        if "window" in v_lower or "sill" in v_lower or "handle" in v_lower:
            groups["Windows"].append(v)
            matched = True

        if not matched:
            groups["Other"].append(v)

    total_printed = 0
    for category, viols in groups.items():
        if viols:
            print(f"\n{category}: {len(viols)}")
            for v in viols:
                print(f"   - {v}")
            total_printed += len(viols)

if __name__ == "__main__":
    test_layout(filename="Input-Layouts/room-layout (1).json")