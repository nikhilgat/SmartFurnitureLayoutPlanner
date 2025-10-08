import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from validator import LayoutValidator
import os

class InteractiveLayoutOptimizer:
    def __init__(self, model_path="./qwen-layout-optimizer-final"):
        print("="*60)
        print("LOADING MODEL - Please wait...")
        print("="*60)
        
        print("\n[1/3] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("[2/3] Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Math-7B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("[3/3] Loading fine-tuned adapter...")
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        print("\n" + "="*60)
        print("MODEL READY!")
        print("="*60 + "\n")
    
    def optimize_layout(self, input_layout):
        """Generate optimized layout from input layout"""
        
        # Create prompt
        messages = [
            {
                "role": "system",
                "content": "You are an expert interior designer specializing in DIN 18040-2 compliant accessible bedroom layouts. Given an input layout, optimize furniture positions to comply with accessibility standards while maintaining livability."
            },
            {
                "role": "user",
                "content": json.dumps(input_layout)
            }
        ]
        
        # Format and tokenize
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs['input_ids'].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only generated tokens
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Parse JSON
        try:
            output_layout = json.loads(response)
            return output_layout, None
        except json.JSONDecodeError as e:
            return None, f"JSON Parse Error: {e}\nRaw output: {response}"
    
    def validate_layout(self, layout):
        """Validate layout and return violations"""
        try:
            validator = LayoutValidator(layout)
            violations = validator.validate()
            return violations
        except Exception as e:
            return [f"Validation error: {e}"]

def main():
    # Load model once
    optimizer = InteractiveLayoutOptimizer()
    
    print("INTERACTIVE MODE")
    print("Commands:")
    print("  - Enter input file path (e.g., 'room-layout.json')")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'help' for commands")
    print()
    
    while True:
        print("-" * 60)
        command = input("Enter input file path (or command): ").strip()
        
        if command.lower() in ['quit', 'exit', 'q']:
            print("\nExiting...")
            break
        
        if command.lower() == 'help':
            print("\nCommands:")
            print("  <filepath>  - Optimize layout from file")
            print("  quit/exit   - Exit program")
            print("  help        - Show this help")
            continue
        
        if not command:
            continue
        
        # Check if file exists
        if not os.path.exists(command):
            print(f"ERROR: File '{command}' not found!")
            continue
        
        # Load input
        try:
            with open(command) as f:
                input_layout = json.load(f)
            print(f"\nOK Loaded input from: {command}")
        except Exception as e:
            print(f"ERROR loading file: {e}")
            continue
        
        # Generate output
        print("Generating optimized layout...")
        output_layout, error = optimizer.optimize_layout(input_layout)
        
        if error:
            print(f"\nERROR: {error}")
            continue
        
        # Generate output filename
        base_name = os.path.splitext(command)[0]
        output_file = f"{base_name}_optimized.json"
        
        # Save output
        with open(output_file, 'w') as f:
            json.dump(output_layout, f, indent=2)
        print(f"OK Saved output to: {output_file}")
        
        # Validate
        violations = optimizer.validate_layout(output_layout)
        print(f"\nValidation: {len(violations)} violations")
        
        if violations:
            print("Violations:")
            for v in violations[:10]:
                print(f"  - {v}")
            if len(violations) > 10:
                print(f"  ... and {len(violations) - 10} more")
        else:
            print("Perfect layout! Zero violations.")
        
        print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")