import json
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch

class FastJSONTrainer:
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
    
    def create_compact_training_data(self, num_examples=200):
        """Create compact training data that's faster to train on."""
        examples = []
        
        # Simplified furniture combinations for faster training
        furniture_combos = [
            ["bed", "wardrobe", "chair"],
            ["bed", "desk", "chair"],
            ["bed", "wardrobe", "table"],
            ["bed", "dresser", "armchair"]
        ]
        
        room_sizes = [(800, 600), (900, 700), (1000, 800)]
        clearances = [74, 81, 88]  # 110cm, 120cm, 130cm
        
        for i in range(num_examples):
            room_w, room_h = random.choice(room_sizes)
            furniture = random.choice(furniture_combos)
            clearance = random.choice(clearances)
            clearance_cm = int(clearance * 1.35)
            
            # Create COMPACT JSON (single line, no spaces)
            input_layout = {
                "room": {"w": room_w, "h": room_h},
                "items": [
                    {"name": f[0].upper(), "x": random.randint(50, room_w-200), 
                     "y": random.randint(50, room_h-200), "w": 100, "h": 100} 
                    for f in furniture
                ]
            }
            
            # Optimized layout with simple rules
            optimized_items = []
            for j, item in enumerate(input_layout["items"]):
                if item["name"] == "B":  # Bed
                    opt_item = {"name": "B", "x": clearance, "y": clearance, "w": 180, "h": 200}
                elif item["name"] == "W":  # Wardrobe
                    opt_item = {"name": "W", "x": room_w-150-clearance, "y": clearance, "w": 150, "h": 60}
                elif item["name"] == "C":  # Chair
                    opt_item = {"name": "C", "x": clearance, "y": room_h-100-clearance, "w": 50, "h": 50}
                else:  # Other furniture
                    opt_item = {"name": item["name"], "x": room_w//2, "y": clearance, "w": 100, "h": 70}
                optimized_items.append(opt_item)
            
            optimized_layout = {
                "room": {"w": room_w, "h": room_h},
                "items": optimized_items
            }
            
            # VERY SHORT prompt format
            prompt = f"Optimize room {room_w}x{room_h}, clearance {clearance_cm}cm:\n{json.dumps(input_layout, separators=(',', ':'))}\nResult:"
            completion = json.dumps(optimized_layout, separators=(',', ':'))
            
            examples.append({"prompt": prompt, "completion": completion})
        
        return examples
    
    def setup_fast_model(self):
        """Setup model with optimized configuration for speed."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16  # Use half precision
        )
        
        # Smaller LoRA configuration for faster training
        lora_config = LoraConfig(
            r=8,  # Reduced from 16
            lora_alpha=16,  # Reduced from 32
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        
        print("Fast model setup complete")
        print(f"Trainable parameters: {self.model.get_nb_trainable_parameters()}")
    
    def format_example_fast(self, batch):
        """Fast tokenization with shorter sequences."""
        texts = [f"{p}\n{c}" for p, c in zip(batch["prompt"], batch["completion"])]
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=256,  # Much shorter sequences
            padding="max_length"
        )
    
    def train_fast(self, num_examples=200, output_dir="llama3.2-3b-fast-json"):
        """Fast training with optimized settings."""
        if not self.model:
            self.setup_fast_model()
        
        # Create compact dataset
        print(f"Creating {num_examples} compact training examples...")
        examples = self.create_compact_training_data(num_examples)
        
        # Save for inspection
        with open("fast_training_data.jsonl", "w") as f:
            for ex in examples[:5]:  # Save first 5 examples
                f.write(json.dumps(ex) + "\n")
        
        print("Example prompt:")
        print(examples[0]["prompt"])
        print("\nExample completion:")
        print(examples[0]["completion"])
        
        # Create dataset
        dataset = Dataset.from_list(examples)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)  # Smaller test set
        
        # Tokenize
        tokenized_dataset = dataset.map(
            self.format_example_fast,
            batched=True,
            remove_columns=["prompt", "completion"]
        )
        
        # Optimized training arguments for speed
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,  # Increased batch size
            gradient_accumulation_steps=2,  # Reduced
            learning_rate=5e-4,  # Higher learning rate for faster convergence
            num_train_epochs=2,  # Fewer epochs
            logging_steps=5,
            save_strategy="epoch",
            fp16=True,
            push_to_hub=False,
            dataloader_num_workers=0,
            remove_unused_columns=True,
            dataloader_pin_memory=False,
            # Speed optimizations
            save_total_limit=1,  # Keep only latest checkpoint
            evaluation_strategy="no",  # Skip evaluation during training
            load_best_model_at_end=False,
            # Memory optimizations
            gradient_checkpointing=True,
            optim="adamw_torch",  # Faster optimizer
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
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train with progress info
        print(f"Starting fast training on {len(examples)} examples...")
        print(f"Estimated training time: 15-30 minutes")
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        trainer.train()
        
        end_time.record()
        torch.cuda.synchronize()
        training_time = start_time.elapsed_time(end_time) / 1000 / 60  # Convert to minutes
        
        print(f"Training completed in {training_time:.1f} minutes")
        
        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Fast model saved to {output_dir}")
        
        return output_dir

class FastInference:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            load_in_4bit=True,
            device_map="auto"
        )
        
        # Load LoRA weights
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
    def optimize_layout_fast(self, room_layout):
        """Fast inference on room layout."""
        # Convert to compact format for model
        compact_input = {
            "room": {"w": room_layout["room"]["width"], "h": room_layout["room"]["height"]},
            "items": [
                {"name": f["name"][0].upper(), "x": f["x"], "y": f["y"], 
                 "w": f["width"], "h": f["height"]} 
                for f in room_layout["furniture"]
            ]
        }
        
        prompt = f"Optimize room {room_layout['room']['width']}x{room_layout['room']['height']}, clearance 110cm:\n{json.dumps(compact_input, separators=(',', ':'))}\nResult:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result_part = response.split("Result:")[-1].strip()
        
        try:
            compact_result = json.loads(result_part)
            
            # Convert back to original format
            optimized_layout = {
                "room": {"width": compact_result["room"]["w"], "height": compact_result["room"]["h"]},
                "furniture": [
                    {"name": self.expand_name(item["name"]), "x": item["x"], "y": item["y"],
                     "width": item["w"], "height": item["h"]}
                    for item in compact_result["items"]
                ]
            }
            
            return optimized_layout
            
        except json.JSONDecodeError:
            print("Model output was not valid JSON, using fallback optimization")
            return self.fallback_optimization(room_layout)
    
    def expand_name(self, short_name):
        """Expand shortened furniture names."""
        name_map = {"B": "Bed", "W": "Wardrobe", "C": "Chair", "D": "Desk", "T": "Table"}
        return name_map.get(short_name, short_name)
    
    def fallback_optimization(self, room_layout):
        """Simple rule-based optimization as fallback."""
        optimized = json.deepcopy(room_layout)
        room_w, room_h = room_layout["room"]["width"], room_layout["room"]["height"]
        clearance = 74
        
        for furniture in optimized["furniture"]:
            name = furniture["name"].lower()
            if "bed" in name:
                furniture["x"], furniture["y"] = clearance, clearance
            elif "wardrobe" in name:
                furniture["x"] = room_w - furniture["width"] - clearance
                furniture["y"] = clearance
            elif "chair" in name:
                furniture["x"] = clearance
                furniture["y"] = room_h - furniture["height"] - clearance
            else:
                furniture["x"] = room_w // 2
                furniture["y"] = clearance
        
        return optimized

def main():
    print("=== Fast JSON Training ===")
    
    trainer = FastJSONTrainer()
    
    # Quick training
    model_path = trainer.train_fast(num_examples=150)  # Even fewer examples
    
    print("\n=== Testing Fast Inference ===")
    
    # Test the trained model
    inference = FastInference(model_path)
    
    # Example room layout
    test_layout = {
        "room": {"width": 800, "height": 600},
        "furniture": [
            {"name": "Bed", "x": 400, "y": 300, "width": 180, "height": 200},
            {"name": "Wardrobe", "x": 100, "y": 100, "width": 150, "height": 60},
            {"name": "Chair", "x": 500, "y": 500, "width": 50, "height": 50}
        ]
    }
    
    print("Original layout:")
    print(json.dumps(test_layout, indent=2))
    
    optimized = inference.optimize_layout_fast(test_layout)
    
    print("\nOptimized layout:")
    print(json.dumps(optimized, indent=2))

if __name__ == "__main__":
    main()