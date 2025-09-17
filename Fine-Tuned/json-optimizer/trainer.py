import json
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import numpy as np
import random

class JSONTrainingDataCreator:
    def __init__(self):
        self.room_sizes = [(800, 600), (900, 700), (1000, 800), (750, 550)]
        self.furniture_types = {
            "beds": [
                {"name": "Bed", "width": 180, "height": 200},
            ],
            "storage": [
                {"name": "Wardrobe", "width": 150, "height": 60},
            ],
            "seating": [
                {"name": "Chair", "width": 50, "height": 50},
                {"name": "Sofa", "width": 200, "height": 90}
            ],
            "tables": [
                {"name": "Bedside Table", "width": 45, "height": 45},
                {"name": "Study Table", "width": 120, "height": 70}
            ]
        }
        self.clearance_requirements = [74, 81, 88, 95]  # 110cm, 120cm, 130cm, 140cm in pixels
    
    def generate_random_layout(self, room_width, room_height, num_furniture=8):
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
        furniture_list.append(bed)
        
        # Add other furniture
        remaining_furniture = [f for f in all_furniture if "bed" not in f["name"].lower()]
        selected_furniture = random.sample(remaining_furniture, min(num_furniture-1, len(remaining_furniture)))
        
        for furniture in selected_furniture:
            furniture_copy = furniture.copy()
            furniture_copy["x"] = random.randint(0, max(0, room_width - furniture_copy["width"]))
            furniture_copy["y"] = random.randint(0, max(0, room_height - furniture_copy["height"]))
            furniture_list.append(furniture_copy)
        
        return furniture_list
    
    def optimize_layout(self, room_width, room_height, furniture_list, min_clearance, openings=None):
        """Enhanced optimization with relations, multiples, collisions, and openings."""
        if openings is None:
            openings = []
        
        # Define avoidance zones (simple buffers around openings)
        avoid_zones = []  # List of (x_min, x_max, y_min, y_max)
        for opening in openings:
            if opening["type"] == "door" and opening["wall"] == "bottom":
                avoid_zones.append((opening["position"] - opening["size"]/2 - min_clearance,
                                    opening["position"] + opening["size"]/2 + min_clearance,
                                    room_height - min_clearance, room_height))
            elif opening["type"] == "window" and opening["wall"] == "left":
                avoid_zones.append((0, min_clearance, opening["position"] - opening["size"]/2,
                                    opening["position"] + opening["size"]/2))
            # Add more for right/top as needed
        
        # Group relations (e.g., study pair)
        pairs = {}
        unpaired = []
        study_table_idx = None
        study_chair_idx = None
        for i, furn in enumerate(furniture_list):
            name_lower = furn["name"].lower()
            if "study table" in name_lower:
                study_table_idx = i
            elif "study chair" in name_lower:
                study_chair_idx = i
            else:
                unpaired.append(i)
        
        if study_table_idx is not None and study_chair_idx is not None:
            pairs[("study_pair", study_table_idx, study_chair_idx)] = True
            unpaired.remove(study_table_idx)
            unpaired.remove(study_chair_idx)
        
        # Base placements (with creativity)
        base_placements = {
            "bed": lambda w, h: (min_clearance, min_clearance) if random.random() > 0.2 else (min_clearance, h // 2),  # 20% east wall
            "wardrobe": lambda w, h: (w - furn["width"] - min_clearance, min_clearance),
            "sofa": lambda w, h: ((w - furn["width"]) // 2, h // 2),
            "study table": lambda w, h: (w // 2 - furn["width"] // 2, min_clearance),
            "study chair": lambda w, h: None,  # Handled in pair
            "bedside table": lambda w, h: (min_clearance + 180 + 10, min_clearance),  # Next to bed
        }
        
        optimized_furniture = [None] * len(furniture_list)
        
        # Place pairs first
        for pair_key, idx1, idx2 in pairs:
            if pair_key == "study_pair":
                table_furn = furniture_list[idx1]
                chair_furn = furniture_list[idx2]
                # Place table, then chair adjacent (right, 10px gap)
                px, py = base_placements["study table"](room_width, room_height)
                # Avoid zones
                while any(px > z[0] and px < z[1] and py > z[2] and py < z[3] for z in avoid_zones):
                    px += 50  # Shift right
                optimized_furniture[idx1] = table_furn.copy(); optimized_furniture[idx1]["x"], optimized_furniture[idx1]["y"] = px, py
                optimized_furniture[idx2] = chair_furn.copy(); 
                optimized_furniture[idx2]["x"], optimized_furniture[idx2]["y"] = px + table_furn["width"] + 10, py
        
        # Place unpaired
        placed_positions = set()  # For collision check
        for i in unpaired:
            furn = furniture_list[i]
            name_lower = furn["name"].lower()
            key = next((k for k in base_placements if k in name_lower), "sofa")  # Default to sofa
            px, py = base_placements[key](room_width, room_height)
            
            # Handle multiples (offset if same name already placed)
            count = sum(1 for f in furniture_list[:i] if f["name"] == furn["name"])
            if count > 0:
                px += count * 100  # Offset right/down
                py += count * 50 if count > 1 else 0
            
            # Avoid zones and collisions
            attempts = 0
            while (attempts < 5 and
                (any(px > z[0] and px < z[1] and py > z[2] and py < z[3] for z in avoid_zones) or
                    any(abs(px - opx) < 50 and abs(py - opy) < 50 for opx, opy in placed_positions))):
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
        
        # Generate random layout
        furniture_list = self.generate_random_layout(room_width, room_height)
        
        # Create optimized version
        optimized_furniture = self.optimize_layout(room_width, room_height, furniture_list, min_clearance)
        
        # Create input and output layouts
        input_layout = {
            "room": {"width": room_width, "height": room_height},
            "furniture": furniture_list
        }
        
        optimized_layout = {
            "room": {"width": room_width, "height": room_height},
            "furniture": optimized_furniture
        }
        
        # Create training prompt
        clearance_cm = int(min_clearance * 1.35)  # Convert pixels to approximate cm
        furniture_names = [f["name"] for f in furniture_list]
        
        prompt = f"""Optimize furniture arrangement in a bedroom for elderly comfort and accessibility.
Room size: {room_width//67}x{room_height//67} ft. Furniture: {', '.join(furniture_names)}. Constraints: {clearance_cm} cm clear paths, wheelchair accessibility.

Input layout JSON:
{json.dumps(input_layout, indent=2)}

Optimized layout JSON:"""
        
        completion = json.dumps(optimized_layout, indent=2)
        
        return {"prompt": prompt, "completion": completion}
    
    def create_dataset(self, num_examples=500):
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
            max_length=1024,  # Increased for JSON content
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
    
    # Create training data
    creator = JSONTrainingDataCreator()
    examples = creator.create_dataset(num_examples=500)
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

if __name__ == "__main__":
    main()