import json
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import numpy as np

class JSONTrainingDataCreator:
    def __init__(self):
        self.room_sizes = [(800, 600), (900, 700), (1000, 800), (750, 550)]
        self.furniture_types = {
            "beds": [
                {"name": "King Bed", "width": 180, "height": 200},
                {"name": "Queen Bed", "width": 160, "height": 200},
                {"name": "Double Bed", "width": 140, "height": 200},
                {"name": "Single Bed", "width": 90, "height": 200}
            ],
            "storage": [
                {"name": "Wardrobe", "width": 150, "height": 60},
                {"name": "Dresser", "width": 120, "height": 50},
                {"name": "Closet", "width": 100, "height": 60}
            ],
            "seating": [
                {"name": "Chair", "width": 50, "height": 50},
                {"name": "Armchair", "width": 80, "height": 80},
                {"name": "Sofa", "width": 200, "height": 90}
            ],
            "tables": [
                {"name": "Bedside Table", "width": 45, "height": 45},
                {"name": "Study Table", "width": 120, "height": 70},
                {"name": "Desk", "width": 100, "height": 60}
            ]
        }
        self.clearance_requirements = [74, 81, 88, 95]  # 110cm, 120cm, 130cm, 140cm in pixels
    
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
    
    def optimize_layout(self, room_width, room_height, furniture_list, min_clearance):
        """Apply optimization rules to create an optimized layout."""
        optimized_furniture = []
        
        for furniture in furniture_list:
            optimized_item = furniture.copy()
            name = furniture["name"].lower()
            
            if "bed" in name:
                # Place bed along north wall with clearance
                optimized_item["x"] = min_clearance
                optimized_item["y"] = min_clearance
                
            elif "wardrobe" in name or "closet" in name or "dresser" in name:
                # Place storage along opposite wall
                optimized_item["x"] = room_width - furniture["width"] - min_clearance
                optimized_item["y"] = min_clearance
                
            elif "chair" in name or "sofa" in name:
                # Place seating with good access
                if "sofa" in name:
                    optimized_item["x"] = (room_width - furniture["width"]) // 2
                    optimized_item["y"] = room_height // 2
                else:
                    optimized_item["x"] = min_clearance
                    optimized_item["y"] = room_height - furniture["height"] - min_clearance
                    
            elif "table" in name or "desk" in name:
                if "bedside" in name:
                    # Place bedside table next to bed
                    optimized_item["x"] = min_clearance + 200  # bed width + small gap
                    optimized_item["y"] = min_clearance
                else:
                    # Place desk/table along wall
                    optimized_item["x"] = room_width // 2 - furniture["width"] // 2
                    optimized_item["y"] = min_clearance
            
            # Ensure furniture stays within room bounds
            optimized_item["x"] = max(min_clearance, min(optimized_item["x"], room_width - furniture["width"] - min_clearance))
            optimized_item["y"] = max(min_clearance, min(optimized_item["y"], room_height - furniture["height"] - min_clearance))
            
            optimized_furniture.append(optimized_item)
        
        return optimized_furniture
    
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