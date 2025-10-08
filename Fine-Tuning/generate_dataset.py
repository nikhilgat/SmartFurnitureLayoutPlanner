import json
import random
import os
import sys
from optimizer import LayoutOptimizer
from validator import LayoutValidator

def generate_random_layout():
    """Generate random input layout"""
    
    # Random room size (400-1000cm width, 300-800cm height)
    room = {
        'width': random.randrange(400, 1000, 50),
        'height': random.randrange(300, 800, 50)
    }
    
    # Random furniture dimensions (realistic ranges)
    furniture = [
        {
            'name': 'Bed',
            'width': random.choice([140, 160, 180, 200]),
            'height': random.choice([190, 200, 210]),
            'zHeight': '55',
            'x': 0, 'y': 0, 'rotation': 0
        },
        {
            'name': 'Wardrobe',
            'width': random.randrange(100, 180, 10),
            'height': random.randrange(50, 70, 10),
            'zHeight': '200',
            'x': 0, 'y': 0, 'rotation': 0
        },
        {
            'name': 'Bedside Table',
            'width': random.randrange(40, 60, 5),
            'height': random.randrange(40, 60, 5),
            'zHeight': '60',
            'x': 0, 'y': 0, 'rotation': 0
        },
        {
            'name': 'Study Table',
            'width': random.randrange(100, 140, 10),
            'height': random.randrange(60, 80, 10),
            'zHeight': '75',
            'x': 0, 'y': 0, 'rotation': 0
        },
        {
            'name': 'Study Chair',
            'width': 50,
            'height': 50,
            'zHeight': '90',
            'x': 0, 'y': 0, 'rotation': 0
        }
    ]
    
    # 70% chance of having sofa
    if random.random() < 0.7:
        furniture.append({
            'name': 'Sofa',
            'width': random.randrange(180, 220, 10),
            'height': random.randrange(80, 100, 10),
            'zHeight': '85',
            'x': 0, 'y': 0, 'rotation': 0
        })
    
    # Random door/window positions
    openings = [
        {
            'type': 'door',
            'wall': random.choice(['top', 'bottom', 'left', 'right']),
            'position': random.randrange(100, 400, 50),
            'size': 90,
            'openingHeight': '210'
        },
        {
            'type': 'window',
            'wall': random.choice(['top', 'bottom', 'left', 'right']),
            'position': random.randrange(100, 400, 50),
            'size': 120,
            'openingHeight': '100',
            'heightFromGround': '90'
        }
    ]
    
    # Randomize initial positions
    for item in furniture:
        item['x'] = random.randrange(0, room['width'] - item['width'], 10)
        item['y'] = random.randrange(0, room['height'] - item['height'], 10)
        item['rotation'] = random.choice([0, 90, 180, 270])
    
    return {
        'room': room,
        'furniture': furniture,
        'openings': openings
    }

def generate_dataset(num_examples=10):
    """Generate training dataset - saves input and output separately"""
    
    # Create output directory
    os.makedirs('training_data', exist_ok=True)
    
    summary = []
    successful = 0
    
    for i in range(num_examples):
        print(f"\n=== Generating example {i+1}/{num_examples} ===")
        
        max_retries = 3
        success = False
        input_layout = None
        optimized_layout = None
        
        for attempt in range(max_retries):
            # Generate random input
            input_layout = generate_random_layout()
            
            # Optimize
            optimizer = LayoutOptimizer(input_layout)
            optimized_layout = optimizer.optimize(max_iterations=5000)
            
            # Check if optimization succeeded
            if optimized_layout is not None:
                success = True
                break
            else:
                print(f"  Attempt {attempt+1} failed, retrying with new layout...")
        
        if not success:
            print(f"  WARNING: Skipping example {i+1} after {max_retries} failed attempts")
            continue
        
        # Save input
        input_file = f'training_data/example_{successful+1:03d}_input.json'
        with open(input_file, 'w') as f:
            json.dump(input_layout, f, indent=2)
        
        # Save output
        output_file = f'training_data/example_{successful+1:03d}_output.json'
        with open(output_file, 'w') as f:
            json.dump(optimized_layout, f, indent=2)
        
        # Validate for summary only
        validator = LayoutValidator(optimized_layout)
        violations = validator.validate()
        
        print(f"OK Saved input:  {input_file}")
        print(f"OK Saved output: {output_file}")
        print(f"  Violations: {len(violations)}")
        
        # Track summary
        summary.append({
            'example': successful+1,
            'input_file': input_file,
            'output_file': output_file,
            'violation_count': len(violations),
            'furniture_count': len(input_layout['furniture']),
            'room_size': f"{input_layout['room']['width']}x{input_layout['room']['height']}"
        })
        
        successful += 1
    
    # Save summary
    with open('training_data/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nOK Generated {successful}/{num_examples} examples in training_data/ folder")
    print(f"OK Summary saved to training_data/summary.json")

if __name__ == '__main__':
    # Get number from command line argument or default to 10
    if len(sys.argv) > 1:
        num_examples = int(sys.argv[1])
    else:
        num_examples = 10
    
    generate_dataset(num_examples=num_examples)