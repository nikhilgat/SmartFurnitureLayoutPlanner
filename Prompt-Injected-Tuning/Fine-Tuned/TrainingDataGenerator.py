"""
Step 3: Training Data Generator for DIN 18040-2 Layout Optimization
Creates training dataset with bad layouts → optimized layouts
"""

import json
import random
import copy
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
import os
from datetime import datetime

# Import from previous steps
from DesignRules import ProfessionalCareHomeRules
from LayoutOptimizer import ProfessionalLayoutOptimizer, Room, Furniture


class TrainingDataGenerator:
    """Generate training data for layout optimization model"""
    
    def __init__(self, rules: ProfessionalCareHomeRules, optimizer: ProfessionalLayoutOptimizer):
        self.rules = rules
        self.optimizer = optimizer
        
        # Furniture templates for generation
        self.furniture_templates = [
            ("Bed", "bed", 180, 200, 55),
            ("Single Bed", "bed", 90, 200, 55),
            ("Double Bed", "bed", 160, 200, 55),
            ("King Bed", "bed", 180, 210, 55),
            ("Wardrobe", "wardrobe", 150, 60, 200),
            ("Closet", "wardrobe", 120, 60, 200),
            ("Large Wardrobe", "wardrobe", 200, 60, 200),
            ("Bedside Table", "bedside_table", 45, 45, 60),
            ("Night Stand", "bedside_table", 50, 40, 55),
            ("Study Table", "desk", 120, 70, 75),
            ("Desk", "desk", 140, 70, 75),
            ("Computer Desk", "desk", 150, 80, 75),
            ("Study Chair", "chair", 50, 50, 90),
            ("Office Chair", "chair", 60, 60, 90),
            ("Armchair", "chair", 80, 80, 85),
            ("Sofa", "sofa", 200, 90, 85),
            ("Loveseat", "sofa", 150, 90, 85),
            ("3-Seater Sofa", "sofa", 220, 100, 85),
            ("Dresser", "dresser", 100, 50, 120),
            ("Chest of Drawers", "dresser", 80, 45, 110),
            ("Coffee Table", "coffee_table", 100, 60, 40),
            ("Side Table", "side_table", 60, 60, 50),
            ("TV Stand", "tv_stand", 120, 40, 60),
            ("Bookshelf", "bookshelf", 80, 30, 180),
            ("Shelf Unit", "shelf", 100, 40, 150),
        ]
        
        # Room size variations
        self.room_sizes = [
            (400, 350),  # Small bedroom
            (450, 400),  # Standard bedroom  
            (500, 450),  # Medium bedroom
            (550, 500),  # Large bedroom
            (600, 500),  # Master bedroom
            (650, 550),  # Extra large
            (700, 600),  # Suite
            (800, 600),  # Luxury suite
        ]
        
        # Door positions
        self.door_positions = ["bottom", "left", "right", "top"]
    
    def generate_random_room(self) -> Room:
        """Generate a random room configuration"""
        width, height = random.choice(self.room_sizes)
        # Add some variation
        width += random.randint(-50, 50)
        height += random.randint(-50, 50)
        
        door_position = random.choice(self.door_positions)
        
        # Calculate door offset based on position
        if door_position in ["bottom", "top"]:
            door_offset = random.randint(100, width - 200)
        else:  # left or right
            door_offset = random.randint(100, height - 200)
        
        return Room(width=width, height=height, 
                   door_position=door_position,
                   door_offset=door_offset)
    
    def generate_furniture_set(self, num_items: int) -> List[Furniture]:
        """Generate a random set of furniture"""
        furniture = []
        
        # Ensure we have at least one bed
        bed_template = random.choice([t for t in self.furniture_templates if "bed" in t[1]])
        furniture.append(Furniture(
            name=bed_template[0],
            type=bed_template[1],
            width=bed_template[2],
            height=bed_template[3],
            z_height=bed_template[4]
        ))
        
        # Add random other furniture
        for i in range(num_items - 1):
            template = random.choice(self.furniture_templates)
            name = template[0]
            
            # Handle duplicate names
            existing_names = [f.name for f in furniture]
            if name in existing_names:
                count = sum(1 for n in existing_names if name in n)
                name = f"{name}_{count + 1}"
            
            furniture.append(Furniture(
                name=name,
                type=template[1],
                width=template[2],
                height=template[3],
                z_height=template[4]
            ))
        
        return furniture
    
    def furniture_to_dict(self, furniture: Furniture) -> Dict:
        """Convert furniture object to dictionary"""
        return {
            "name": furniture.name,
            "type": furniture.type,
            "width": furniture.width,
            "height": furniture.height,
            "x": furniture.x,
            "y": furniture.y,
            "zHeight": furniture.z_height,
            "rotation": furniture.rotation
        }
    
    def room_to_dict(self, room: Room) -> Dict:
        """Convert room object to dictionary"""
        return {
            "width": room.width,
            "height": room.height,
            "door_position": room.door_position,
            "door_offset": room.door_offset
        }
    
    def create_openings(self, room: Room) -> List[Dict]:
        """Create door and window openings for a room"""
        openings = []
        
        # Add door
        if room.door_position == "bottom":
            openings.append({
                "type": "door",
                "x": room.door_offset,
                "y": room.height,
                "size": 90,
                "openingHeight": "210"
            })
        elif room.door_position == "top":
            openings.append({
                "type": "door",
                "x": room.door_offset,
                "y": 0,
                "size": 90,
                "openingHeight": "210"
            })
        elif room.door_position == "left":
            openings.append({
                "type": "door",
                "x": 0,
                "y": room.door_offset,
                "size": 90,
                "openingHeight": "210"
            })
        else:  # right
            openings.append({
                "type": "door",
                "x": room.width,
                "y": room.door_offset,
                "size": 90,
                "openingHeight": "210"
            })
        
        # Add 1-2 random windows
        num_windows = random.randint(1, 2)
        used_walls = [room.door_position]
        
        for _ in range(num_windows):
            available_walls = [w for w in self.door_positions if w not in used_walls]
            if not available_walls:
                break
            
            wall = random.choice(available_walls)
            used_walls.append(wall)
            
            if wall == "bottom":
                openings.append({
                    "type": "window",
                    "x": random.randint(100, room.width - 200),
                    "y": room.height,
                    "size": 120,
                    "openingHeight": "100",
                    "heightFromGround": "90"
                })
            elif wall == "top":
                openings.append({
                    "type": "window",
                    "x": random.randint(100, room.width - 200),
                    "y": 0,
                    "size": 120,
                    "openingHeight": "100",
                    "heightFromGround": "90"
                })
            elif wall == "left":
                openings.append({
                    "type": "window",
                    "x": 0,
                    "y": random.randint(100, room.height - 200),
                    "size": 120,
                    "openingHeight": "100",
                    "heightFromGround": "90"
                })
            else:  # right
                openings.append({
                    "type": "window",
                    "x": room.width,
                    "y": random.randint(100, room.height - 200),
                    "size": 120,
                    "openingHeight": "100",
                    "heightFromGround": "90"
                })
        
        return openings
    
    def generate_training_example(self, wheelchair_mode: bool = True) -> Dict:
        """Generate a single training example"""
        # Generate room
        room = self.generate_random_room()
        
        # Generate furniture (3-7 items)
        num_items = random.randint(3, 7)
        furniture = self.generate_furniture_set(num_items)
        
        # Create bad layout
        bad_layout = self.optimizer.generate_bad_layout(room, furniture)
        
        # Optimize layout
        optimized_layout, metadata = self.optimizer.optimize_layout(
            room, furniture, wheelchair=wheelchair_mode
        )
        
        # Create openings
        openings = self.create_openings(room)
        
        # Build training example
        example = {
            # Input (bad layout)
            "input": json.dumps({
                "room": self.room_to_dict(room),
                "furniture": [self.furniture_to_dict(f) for f in bad_layout],
                "openings": openings,
                "requirements": {
                    "wheelchair_user": wheelchair_mode,
                    "min_clearance": 150 if wheelchair_mode else 120,
                    "turning_space": "150x150" if wheelchair_mode else "120x120",
                    "pathway_width": 150 if wheelchair_mode else 120
                }
            }, separators=(',', ':')),
            
            # Output (optimized layout)
            "output": json.dumps({
                "room": self.room_to_dict(room),
                "furniture": [self.furniture_to_dict(f) for f in optimized_layout],
                "openings": openings,
                "metadata": {
                    "din_compliant": metadata["din_18040_2_compliant"],
                    "wheelchair_optimized": metadata["wheelchair_optimized"],
                    "all_furniture_placed": metadata["all_furniture_placed"],
                    "placement_strategy": metadata["placement_stats"]
                }
            }, separators=(',', ':')),
            
            # Instruction for training
            "instruction": self.generate_instruction(wheelchair_mode),
            
            # Metadata
            "metadata": {
                "room_size": f"{room.width}x{room.height}",
                "furniture_count": len(furniture),
                "wheelchair_mode": wheelchair_mode,
                "placement_stats": metadata["placement_stats"],
                "generation_timestamp": datetime.now().isoformat()
            }
        }
        
        return example
    
    def generate_instruction(self, wheelchair_mode: bool) -> str:
        """Generate varied instructions for training"""
        if wheelchair_mode:
            instructions = [
                "Optimize this room layout for wheelchair accessibility following DIN 18040-2 standards. Ensure 150cm clearances and turning spaces.",
                "Rearrange this bedroom furniture for wheelchair users according to DIN 18040-2. Maintain 150cm pathways and maneuvering space.",
                "Apply DIN 18040-2 wheelchair requirements to optimize this furniture layout. All items must be accessible with 150cm clearances.",
                "Transform this layout to meet DIN 18040-2 accessibility standards for wheelchair users. Preserve all furniture while ensuring proper clearances.",
                "Reorganize this room following DIN 18040-2 guidelines for wheelchair accessibility. Create 150x150cm turning spaces and maintain clear pathways.",
            ]
        else:
            instructions = [
                "Optimize this room layout following DIN 18040-2 standards. Ensure proper clearances and circulation paths.",
                "Rearrange this bedroom furniture according to DIN 18040-2 requirements. Maintain minimum 120cm pathways.",
                "Apply DIN 18040-2 standards to improve this furniture arrangement. All items must remain accessible.",
                "Transform this layout to meet DIN 18040-2 accessibility guidelines. Keep all furniture while ensuring proper spacing.",
                "Reorganize this room following DIN 18040-2 principles. Create functional zones with adequate clearances.",
            ]
        
        return random.choice(instructions)
    
    def generate_dataset(self, num_examples: int = 2000, wheelchair_ratio: float = 0.5) -> List[Dict]:
        """Generate complete training dataset"""
        dataset = []
        num_wheelchair = int(num_examples * wheelchair_ratio)
        
        print(f"Generating {num_examples} training examples...")
        print(f"  - Wheelchair accessible: {num_wheelchair}")
        print(f"  - Standard accessible: {num_examples - num_wheelchair}")
        
        # Generate wheelchair examples
        for i in range(num_wheelchair):
            if i % 100 == 0:
                print(f"  Generated {i}/{num_examples} examples...")
            
            example = self.generate_training_example(wheelchair_mode=True)
            dataset.append(example)
        
        # Generate standard examples
        for i in range(num_wheelchair, num_examples):
            if i % 100 == 0:
                print(f"  Generated {i}/{num_examples} examples...")
            
            example = self.generate_training_example(wheelchair_mode=False)
            dataset.append(example)
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        print(f"✔ Generated {len(dataset)} training examples")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str = "din_training_dataset.jsonl"):
        """Save dataset in JSONL format (one example per line)"""
        with open(filename, 'w', encoding='utf-8') as f:
            for example in dataset:
                # Write each example as a single line
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"✔ Dataset saved to {filename}")
        print(f"  File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    def save_dataset_alpaca_format(self, dataset: List[Dict], 
                                   filename: str = "din_training_alpaca.json"):
        """Save dataset in Alpaca format for fine-tuning"""
        alpaca_dataset = []
        
        for example in dataset:
            alpaca_dataset.append({
                "instruction": example["instruction"],
                "input": example["input"],
                "output": example["output"]
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(alpaca_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✔ Alpaca format saved to {filename}")
    
    def validate_dataset(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Validate dataset quality and statistics"""
        stats = {
            "total_examples": len(dataset),
            "wheelchair_examples": sum(1 for ex in dataset if ex["metadata"]["wheelchair_mode"]),
            "standard_examples": sum(1 for ex in dataset if not ex["metadata"]["wheelchair_mode"]),
            "avg_furniture_count": sum(ex["metadata"]["furniture_count"] for ex in dataset) / len(dataset),
            "room_sizes": {},
            "placement_strategies": {
                "zone_success": 0,
                "wall_success": 0,
                "grid_success": 0,
                "fallback_used": 0
            }
        }
        
        # Aggregate placement strategies
        for example in dataset:
            for strategy, count in example["metadata"]["placement_stats"].items():
                stats["placement_strategies"][strategy] += count
        
        # Count room sizes
        for example in dataset:
            size = example["metadata"]["room_size"]
            stats["room_sizes"][size] = stats["room_sizes"].get(size, 0) + 1
        
        return stats


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("STEP 3: TRAINING DATA GENERATION")
    print("="*60)
    
    # Initialize components
    print("\nInitializing components...")
    rules = ProfessionalCareHomeRules()
    optimizer = ProfessionalLayoutOptimizer(rules)
    generator = TrainingDataGenerator(rules, optimizer)
    
    # Generate dataset
    print("\nGenerating training dataset...")
    dataset = generator.generate_dataset(
        num_examples=2000,
        wheelchair_ratio=0.5  # 50% wheelchair, 50% standard
    )
    
    # Save datasets
    print("\nSaving datasets...")
    generator.save_dataset(dataset, "din_training_dataset.jsonl")
    generator.save_dataset_alpaca_format(dataset, "din_training_alpaca.json")
    
    # Validate and show statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    stats = generator.validate_dataset(dataset)
    print(f"Total examples: {stats['total_examples']}")
    print(f"Wheelchair accessible: {stats['wheelchair_examples']} ({100*stats['wheelchair_examples']/stats['total_examples']:.1f}%)")
    print(f"Standard accessible: {stats['standard_examples']} ({100*stats['standard_examples']/stats['total_examples']:.1f}%)")
    print(f"Average furniture per room: {stats['avg_furniture_count']:.1f}")
    
    print("\n" + "="*60)
    print("PLACEMENT STRATEGY USAGE")
    print("="*60)
    
    total_placements = sum(stats['placement_strategies'].values())
    for strategy, count in stats['placement_strategies'].items():
        percentage = 100 * count / total_placements if total_placements > 0 else 0
        print(f"{strategy}: {count} ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("STEP 3 COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - din_training_dataset.jsonl (for training pipeline)")
    print("  - din_training_alpaca.json (Alpaca format)")
    print("\nNext: Run Step 4 to train the model with this data")