
import json
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np

class JSONTrainingDataCreator:
    def __init__(self):
        self.room_sizes = [(800, 600), (900, 700), (1000, 800), (750, 550)]
        self.furniture_types = {
            "beds": [
                {"name": "Bed", "width": 180, "height": 200}
            ],
            "storage": [
                {"name": "Wardrobe", "width": 150, "height": 60}
            ],
            "seating": [
                {"name": "Sofa", "width": 200, "height": 90}
            ],
            "tables": [
                {"name": "Bedside Table", "width": 45, "height": 45},
                {"name": "Study Chair", "width": 50, "height": 50},
                {"name": "Study Table", "width": 120, "height": 70},
            ]
        }
        self.clearance_requirements = [74, 81, 88, 95]  # 110cm, 120cm, 130cm, 140cm in pixels
        self.opening_types = ["door", "window"]
        self.walls = ["bottom", "left", "right", "top"]  # Common walls for bedrooms
        # zHeight defaults by type
        self.zheights = {
            "bed": "55",
            "wardrobe": "200",
            "study chair": "90",
            "sofa": "85",
            "bedside table": "60",
            "study table": "75",
        }
    
    def generate_random_layout(self, room_width, room_height, num_furniture=4):
        """Generate a random (potentially suboptimal) furniture layout."""
        furniture_list = []
        all_furniture = []
        
        # Collect all furniture types
        for category in self.furniture_types.values():
            all_furniture.extend(category)
        
        # Always include a bed
        bed = random.choice(self.furniture_types["beds"]).copy()
        bed["x"] = random.randint(0, max(0, room_width - bed["width"]))
        bed["y"] = random.randint(0, max(0, room_height - bed["height"]))
        bed["zHeight"] = self.zheights["bed"]
        bed["rotation"] = random.choice([0, 90, 180, 270])
        furniture_list.append(bed)
        
        # Add other furniture
        remaining_furniture = [f for f in all_furniture if "bed" not in f["name"].lower()]
        selected_furniture = random.sample(remaining_furniture, min(num_furniture-1, len(remaining_furniture)))
        
        for furniture in selected_furniture:
            furniture_copy = furniture.copy()
            furniture_copy["x"] = random.randint(0, max(0, room_width - furniture_copy["width"]))
            furniture_copy["y"] = random.randint(0, max(0, room_height - furniture_copy["height"]))
            # Set zHeight and rotation
            key = next((k for k in self.zheights if k in furniture_copy["name"].lower()), "sofa")
            furniture_copy["zHeight"] = self.zheights[key]
            furniture_copy["rotation"] = random.choice([0, 90, 180, 270])
            furniture_list.append(furniture_copy)
        
        return furniture_list
    
    def generate_random_openings(self, room_width, room_height):
        """Generate 1 door + 1-2 windows."""
        openings = []
        # Always add a door (e.g., bottom wall, centered)
        door_pos = room_width // 2
        openings.append({
            "type": "door",
            "wall": random.choice(["bottom", "top"]),  # Vary door wall
            "position": door_pos,
            "size": random.randint(80, 100),
            "openingHeight": "210"
        })
        # Add 1-2 windows
        num_windows = random.randint(1, 2)
        for _ in range(num_windows):
            wall = random.choice(self.walls)
            if wall in ["left", "right"]:
                pos = random.randint(100, min(room_width, room_height) - 100)
            else:
                pos = random.randint(100, room_width - 100)
            openings.append({
                "type": "window",
                "wall": wall,
                "position": pos,
                "size": random.randint(100, 150),
                "openingHeight": "100",
                "heightFromGround": "90"
            })
        return openings
    
    def optimize_layout(self, room_width, room_height, furniture_list, min_clearance, openings=None):
        """Enhanced optimization with relations, multiples, collisions, and openings."""
        if openings is None:
            openings = []
        
        # Define avoidance zones (simple buffers around openings)
        avoid_zones = []  # List of (x_min, x_max, y_min, y_max)
        for opening in openings:
            if opening["type"] == "door" and opening["wall"] == "bottom":
                avoid_zones.append((
                    opening["position"] - opening["size"]/2 - min_clearance,
                    opening["position"] + opening["size"]/2 + min_clearance,
                    room_height - min_clearance, room_height
                ))
            elif opening["type"] == "window" and opening["wall"] == "left":
                avoid_zones.append((
                    0, min_clearance,
                    opening["position"] - opening["size"]/2,
                    opening["position"] + opening["size"]/2
                ))
            elif opening["type"] == "window" and opening["wall"] == "right":
                avoid_zones.append((
                    room_width - min_clearance, room_width,
                    opening["position"] - opening["size"]/2,
                    opening["position"] + opening["size"]/2
                ))
            # Add more for top as needed
        
        # Group relations (e.g., study pair)
        pairs = {}
        unpaired = list(range(len(furniture_list)))
        study_table_idx = None
        study_chair_idx = None
        for i in unpaired:
            furn = furniture_list[i]
            name_lower = furn["name"].lower()
            if "study table" in name_lower:
                study_table_idx = i
            elif "study chair" in name_lower:
                study_chair_idx = i
        
        if study_table_idx is not None and study_chair_idx is not None:
            pairs[("study_pair", study_table_idx, study_chair_idx)] = True
            unpaired.remove(study_table_idx)
            unpaired.remove(study_chair_idx)
        
        # Base placements (with creativity)
        def get_base_placement(key, room_width, room_height, furn):
            if key == "bed":
                if random.random() > 0.2:
                    return min_clearance, min_clearance  # North wall
                else:
                    return min_clearance, room_height // 2  # East wall variant
            elif "wardrobe" in key or "closet" in key or "dresser" in key:
                return room_width - furn["width"] - min_clearance, min_clearance
            elif "sofa" in key:
                return (room_width - furn["width"]) // 2, room_height // 2
            elif "study table" in key:
                return room_width // 2 - furn["width"] // 2, min_clearance
            elif "bedside table" in key:
                return min_clearance + 180 + 10, min_clearance  # Next to bed
            else:
                return min_clearance, room_height - furn["height"] - min_clearance  # Default chair-like
        
        optimized_furniture = [None] * len(furniture_list)
        
        # Place pairs first
        for pair_key, idx1, idx2 in pairs.keys():
            if pair_key == "study_pair":
                table_furn = furniture_list[idx1]
                chair_furn = furniture_list[idx2]
                # Place table, then chair adjacent (right, 10px gap)
                px, py = get_base_placement("study table", room_width, room_height, table_furn)
                # Avoid zones
                attempts = 0
                while (attempts < 5 and
                       any(px > z[0] and px < z[1] and py > z[2] and py < z[3] for z in avoid_zones)):
                    px += 50  # Shift right
                    attempts += 1
                optimized_furniture[idx1] = table_furn.copy()
                optimized_furniture[idx1]["x"], optimized_furniture[idx1]["y"] = px, py
                optimized_furniture[idx2] = chair_furn.copy()
                optimized_furniture[idx2]["x"], optimized_furniture[idx2]["y"] = px + table_furn["width"] + 10, py
        
        # Place unpaired
        placed_positions = set()  # For collision check
        for i in unpaired:
            furn = furniture_list[i]
            name_lower = furn["name"].lower()
            key = next((k for k in ["bed", "wardrobe", "sofa", "study table", "bedside table"] if k in name_lower), "chair")
            px, py = get_base_placement(key, room_width, room_height, furn)
            
            # Handle multiples (offset if same name already placed)
            count = sum(1 for f in furniture_list[:i] if f["name"] == furn["name"])
            if count > 0:
                px += count * 100  # Offset right/down
                py += count * 50 if count > 1 else 0
            
            # Avoid zones and collisions (simple overlap check)
            attempts = 0
            while (attempts < 5 and
                   (any(px > z[0] and px < z[1] and py > z[2] and py < z[3] for z in avoid_zones) or
                    any(abs(px - opx) < furn["width"] + 50 and abs(py - opy) < furn["height"] + 50 for opx, opy in placed_positions))):
                px += random.randint(-50, 50)
                py += random.randint(-50, 50)
                attempts += 1
            
            # Bounds
            px = max(min_clearance, min(px, room_width - furn["width"] - min_clearance))
            py = max(min_clearance, min(py, room_height - furn["height"] - min_clearance))
            
            optimized_furniture[i] = furn.copy()
            optimized_furniture[i]["x"], optimized_furniture[i]["y"] = px, py
            placed_positions.add((px, py))
        
        return [f for f in optimized_furniture if f is not None]
    
    def create_training_example(self):
        """Create a single training example with input and optimized layouts."""
        # Random room size
        room_width, room_height = random.choice(self.room_sizes)
        min_clearance = random.choice(self.clearance_requirements)
        num_furniture = random.randint(3, 6)
        
        # Generate random openings
        openings = self.generate_random_openings(room_width, room_height)
        
        # Generate random layout
        furniture_list = self.generate_random_layout(room_width, room_height, num_furniture)
        
        # Create input and output layouts
        input_layout = {
            "room": {"width": room_width, "height": room_height},
            "furniture": furniture_list,
            "openings": openings
        }
        
        # Create optimized version (single variant for simplicity; expand if needed)
        styles = ["cozy", "spacious", "minimal"]
        style = random.choice(styles)
        var_clearance = min_clearance + random.randint(-10, 10)
        optimized_furniture = self.optimize_layout(room_width, room_height, furniture_list, var_clearance, openings)
        optimized_layout = {
            "room": {"width": room_width, "height": room_height},
            "furniture": optimized_furniture,
            "openings": openings
        }
        
        # Create training prompt
        clearance_cm = int(min_clearance * 1.35)  # Convert pixels to approximate cm
        furniture_names = [f["name"] for f in furniture_list]
        
        prompt = f"""Be creative: Optimize furniture arrangement in a {style} bedroom for elderly comfort and accessibility, pairing related items (e.g., study chair with table, bedside with bed) and avoiding openings.
Room size: {room_width//67}x{room_height//67} ft. Furniture: {', '.join(furniture_names)}. Constraints: {clearance_cm} cm clear paths, wheelchair accessibility.

Input layout JSON:
{json.dumps(input_layout, indent=2)}

Optimized layout JSON:"""
        
        completion = json.dumps(optimized_layout, indent=2)
        
        return {"prompt": prompt, "completion": completion}
    
    def create_dataset(self, num_examples=1000):
        """Create a full training dataset."""
        examples = []
        for _ in range(num_examples):
            example = self.create_training_example()
            examples.append(example)
        return examples
    
    def save_dataset(self, examples, filename="json_furniture_dataset.jsonl"):
        """Save the dataset to a JSONL file."""
        with open(filename, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        print(f"Dataset saved to {filename}")

class JSONModelTrainer:
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
    
    def setup_model(self):
        """Setup tokenizer and model for training."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            load_in_4bit=True,
            device_map="auto"
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("Model setup complete")
    
    def format_example(self, batch):
        """Format examples for training."""
        texts = [f"{p}\n{c}" for p, c in zip(batch["prompt"], batch["completion"])]
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=2048,  # Increased for JSON content
            padding="max_length"
        )
    
    def train_model(self, dataset_file, output_dir="llama3.2-3b-json-furniture"):
        """Train the model on JSON furniture data."""
        if not self.model or not self.tokenizer:
            self.setup_model()
        
        # Load dataset
        dataset = Dataset.from_json(dataset_file)
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.format_example,
            batched=True,
            remove_columns=["prompt", "completion"]
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,  # Reduced due to longer sequences
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            warmup_ratio=0.1,
            fp16=True,
            push_to_hub=False,
            dataloader_num_workers=0
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

def main():
    print("Creating JSON-based training dataset...")
    
    creator = JSONTrainingDataCreator()
    examples = creator.create_dataset(num_examples=1000)
    creator.save_dataset(examples, "json_furniture_dataset.jsonl")
    
    print(f"Created {len(examples)} training examples")
    
    # Show example
    print("\nExample training data:")
    print("PROMPT:")
    print(examples[0]["prompt"])
    print("\nCOMPLETION:")
    print(examples[0]["completion"])
    
    # Optionally train the model
    train_now = input("\nDo you want to start training now? (y/n): ")
    if train_now.lower() == 'y':
        trainer = JSONModelTrainer()
        trainer.train_model("json_furniture_dataset.jsonl")
        
        # Optional quick inference test (uncomment and provide sample_input_json path)
        # sample_input = json.load(open("room-layout-1.json"))
        # input_text = f"""Be creative: Optimize furniture arrangement in a bedroom for elderly comfort and accessibility.
        # Room size: {sample_input['room']['width']//67}x{sample_input['room']['height']//67} ft. Furniture: {', '.join([f['name'] for f in sample_input['furniture']])}. Constraints: 110 cm clear paths, wheelchair accessibility.
        # Input layout JSON:
        # {json.dumps(sample_input, indent=2)}
        # Optimized layout JSON:"""
        # inputs = trainer.tokenizer(input_text, return_tensors="pt")
        # outputs = trainer.model.generate(**inputs, max_new_tokens=500, temperature=0.8)
        # print("Inference Output:")
        # print(trainer.tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
