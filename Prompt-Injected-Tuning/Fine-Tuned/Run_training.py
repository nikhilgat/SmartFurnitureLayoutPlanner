"""
Step 4: Model Training Script for DIN 18040-2 Layout Optimization
Trains LLaMA-3.2-3B on the generated dataset
Optimized for RTX 4070 8GB VRAM
"""

import json
import torch
import gc
import os
from typing import Dict, List
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class DINModelTrainer:
    """Train LLaMA model for DIN-compliant layout optimization"""
    
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check GPU
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"   Current Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        else:
            print("⚠️ WARNING: No GPU detected! Training will be extremely slow")
    
    def clear_memory(self):
        """Clear GPU memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def load_and_prepare_model(self):
        """Load model with 4-bit quantization for 8GB VRAM"""
        print("\n" + "="*60)
        print("LOADING MODEL")
        print("="*60)
        
        # Clear memory first
        self.clear_memory()
        
        # Load tokenizer
        print(f"Loading tokenizer: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 4-bit quantization config for 8GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with quantization
        print(f"Loading model with 4-bit quantization...")
        print("This may take a few minutes...")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("1. Ensure you have the model files downloaded")
            print("2. Check model path is correct")
            print("3. Try: pip install bitsandbytes accelerate")
            return False
        
        # Disable cache to save memory
        self.model.config.use_cache = False
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA for efficient training
        print("\nConfiguring LoRA for efficient training...")
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        if torch.cuda.is_available():
            print(f"\n✅ Model loaded successfully")
            print(f"   VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        return True
    
    def load_dataset(self, dataset_file: str = "din_training_dataset.jsonl"):
        """Load and prepare the dataset"""
        print("\n" + "="*60)
        print("LOADING DATASET")
        print("="*60)
        
        if not os.path.exists(dataset_file):
            print(f"❌ Dataset file not found: {dataset_file}")
            print("Please run Step3_TrainingDataGenerator.py first")
            return None
        
        print(f"Loading {dataset_file}...")
        examples = []
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error parsing line {line_num}: {e}")
                    continue
        
        print(f"✅ Loaded {len(examples)} training examples")
        
        if len(examples) == 0:
            print("❌ No valid examples found in dataset")
            return None
        
        return examples
    
    def format_training_text(self, example: Dict) -> str:
        """Format example for instruction-following training"""
        # Using Alpaca-style format
        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        
        return text
    
    def prepare_dataset_for_training(self, examples: List[Dict]):
        """Convert examples to HuggingFace dataset format"""
        print("\n" + "="*60)
        print("PREPARING DATASET FOR TRAINING")
        print("="*60)
        
        # Format all examples
        formatted_examples = []
        for example in examples:
            text = self.format_training_text(example)
            formatted_examples.append({"text": text})
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(formatted_examples)
        
        # Split into train/test
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        print(f"Dataset split:")
        print(f"  - Training: {len(dataset['train'])} examples")
        print(f"  - Testing: {len(dataset['test'])} examples")
        
        # Tokenize the dataset
        print("\nTokenizing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing"
        )
        
        print("✅ Dataset prepared for training")
        return tokenized_dataset
    
    def train(self, tokenized_dataset, output_dir: str = "din_llama_finetuned"):
        """Train the model"""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Training arguments optimized for 8GB VRAM
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # Training hyperparameters
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Must be 1 for 8GB VRAM
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Effective batch size = 8
            
            # Learning rate
            learning_rate=2e-4,
            warmup_steps=50,
            
            # Mixed precision for memory efficiency
            fp16=True,
            
            # Logging and saving
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            
            # Memory optimization
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",  # 8-bit optimizer
            
            # Other settings
            report_to="none",  # Disable wandb/tensorboard
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
        )
        
        print("\nTraining Configuration:")
        print(f"  • Epochs: {training_args.num_train_epochs}")
        print(f"  • Batch size: {training_args.per_device_train_batch_size}")
        print(f"  • Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"  • Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"  • Learning rate: {training_args.learning_rate}")
        print(f"  • Output directory: {output_dir}")
        
        print("\n🚀 Starting training...")
        print("This will take 2-3 hours. You'll see progress updates every 10 steps.")
        print("Press Ctrl+C to safely stop and save checkpoint.\n")
        
        # Train
        try:
            trainer.train()
            
            # Save final model
            print("\n" + "="*60)
            print("SAVING MODEL")
            print("="*60)
            
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            print(f"✅ Model saved to {output_dir}/")
            
            # Save training info
            training_info = {
                "model_id": self.model_id,
                "dataset_size": len(tokenized_dataset["train"]),
                "epochs": training_args.num_train_epochs,
                "lora_rank": 16,
                "quantization": "4-bit",
                "final_loss": trainer.state.log_history[-1].get("loss", "N/A")
            }
            
            with open(f"{output_dir}/training_info.json", "w") as f:
                json.dump(training_info, f, indent=2)
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Training interrupted by user")
            print("Saving checkpoint...")
            trainer.save_model(f"{output_dir}_checkpoint")
            self.tokenizer.save_pretrained(f"{output_dir}_checkpoint")
            print(f"Checkpoint saved to {output_dir}_checkpoint/")
            return False
        
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            return False
    
    def test_model(self, model_path: str = "din_llama_finetuned"):
        """Test the trained model with a sample input"""
        print("\n" + "="*60)
        print("TESTING TRAINED MODEL")
        print("="*60)
        
        # Test input
        test_input = {
            "room": {"width": 500, "height": 400, "door_position": "bottom", "door_offset": 200},
            "furniture": [
                {"name": "Bed", "type": "bed", "width": 180, "height": 200, "x": 50, "y": 50},
                {"name": "Wardrobe", "type": "wardrobe", "width": 150, "height": 60, "x": 300, "y": 100},
                {"name": "Desk", "type": "desk", "width": 120, "height": 70, "x": 100, "y": 250}
            ],
            "openings": [{"type": "door", "x": 200, "y": 400, "size": 90}],
            "requirements": {"wheelchair_user": True, "min_clearance": 150}
        }
        
        instruction = "Optimize this room layout for wheelchair accessibility following DIN 18040-2 standards."
        
        # Format prompt
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{json.dumps(test_input, separators=(',', ':'))}

### Response:
"""
        
        print("Test input:")
        print(json.dumps(test_input, indent=2))
        
        print("\nGenerating optimized layout...")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        print("\nModel output:")
        print(response[:500] + "..." if len(response) > 500 else response)
        print("\n✅ Test complete!")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("DIN 18040-2 LAYOUT OPTIMIZER - MODEL TRAINING")
    print("="*60)
    print("\nStep 4: Train LLaMA-3.2-3B on DIN compliance dataset")
    print("Optimized for RTX 4070 8GB VRAM")
    
    # Initialize trainer
    trainer = DINModelTrainer()
    
    # Check dataset exists
    if not os.path.exists("din_training_dataset.jsonl"):
        print("\n❌ Dataset not found!")
        print("Please run Step3_TrainingDataGenerator.py first")
        exit(1)
    
    # Load and prepare model
    print("\n" + "-"*60)
    if not trainer.load_and_prepare_model():
        print("Failed to load model. Exiting.")
        exit(1)
    
    # Load dataset
    print("\n" + "-"*60)
    examples = trainer.load_dataset("din_training_dataset.jsonl")
    if examples is None:
        print("Failed to load dataset. Exiting.")
        exit(1)
    
    # Prepare dataset for training
    print("\n" + "-"*60)
    tokenized_dataset = trainer.prepare_dataset_for_training(examples)
    
    # Confirm before starting long training
    print("\n" + "="*60)
    print("READY TO START TRAINING")
    print("="*60)
    print("\nEstimated time: 2-3 hours on RTX 4070")
    print("The model will be saved to: ./din_llama_finetuned/")
    
    response = input("\n🚀 Start training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled by user")
        exit(0)
    
    # Train model
    if trainer.train(tokenized_dataset):
        # Test the trained model
        trainer.test_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\nYour trained model is saved in: ./din_llama_finetuned/")
        print("\nNext steps:")
        print("1. Use the model for inference on your layouts")
        print("2. Create an inference wrapper (Step 5)")
        print("3. Integrate with your UI")
    else:
        print("\nTraining did not complete successfully")
        print("Check the checkpoint directory if training was interrupted")