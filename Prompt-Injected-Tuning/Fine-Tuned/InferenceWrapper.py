"""
Step 5: Inference Wrapper for DIN 18040-2 Layout Optimizer
Use the trained model to optimize room layouts
"""

import json
import torch
import gc
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")


class DINLayoutOptimizer:
    """Production inference wrapper for the trained DIN layout optimization model"""
    
    def __init__(self, model_path: str = "Prompt-Injected-Tuning/Fine-Tuned/din_llama_finetuned"):
        """
        Initialize the optimizer with the trained model
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("="*60)
        print("DIN 18040-2 LAYOUT OPTIMIZER")
        print("="*60)
        print(f"Model path: {model_path}")
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        print("\nLoading model...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with automatic device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            
            if torch.cuda.is_available():
                print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\nMake sure you have run Step4_ModelTraining.py first")
            return False
    
    def validate_input(self, layout_data: Dict) -> tuple[bool, str]:
        """
        Validate the input layout data
        
        Returns:
            (is_valid, error_message)
        """
        # Check required fields
        if "room" not in layout_data:
            return False, "Missing 'room' field"
        
        if "furniture" not in layout_data:
            return False, "Missing 'furniture' field"
        
        # Validate room
        room = layout_data["room"]
        if "width" not in room or "height" not in room:
            return False, "Room must have 'width' and 'height'"
        
        # Validate furniture
        furniture = layout_data["furniture"]
        if not isinstance(furniture, list):
            return False, "Furniture must be a list"
        
        for i, item in enumerate(furniture):
            required_fields = ["name", "width", "height"]
            for field in required_fields:
                if field not in item:
                    return False, f"Furniture item {i} missing '{field}'"
        
        return True, ""
    
    def prepare_input(self, layout_data: Dict, wheelchair_mode: bool = True) -> Dict:
        """
        Prepare and standardize input data
        """
        # Ensure all furniture has required fields
        for item in layout_data["furniture"]:
            if "type" not in item:
                # Infer type from name
                name_lower = item["name"].lower()
                if "bed" in name_lower:
                    item["type"] = "bed"
                elif "desk" in name_lower or "table" in name_lower:
                    item["type"] = "desk"
                elif "wardrobe" in name_lower or "closet" in name_lower:
                    item["type"] = "wardrobe"
                elif "chair" in name_lower:
                    item["type"] = "chair"
                elif "sofa" in name_lower:
                    item["type"] = "sofa"
                else:
                    item["type"] = "furniture"
            
            # Add default positions if missing
            if "x" not in item:
                item["x"] = 0
            if "y" not in item:
                item["y"] = 0
            if "zHeight" not in item:
                item["zHeight"] = 100
            if "rotation" not in item:
                item["rotation"] = 0
        
        # Add room defaults
        if "door_position" not in layout_data["room"]:
            layout_data["room"]["door_position"] = "bottom"
        if "door_offset" not in layout_data["room"]:
            layout_data["room"]["door_offset"] = layout_data["room"]["width"] // 3
        
        # Add openings if missing
        if "openings" not in layout_data:
            room = layout_data["room"]
            layout_data["openings"] = [{
                "type": "door",
                "x": room["door_offset"] if room["door_position"] in ["bottom", "top"] else 0,
                "y": room["height"] if room["door_position"] == "bottom" else 0,
                "size": 90,
                "openingHeight": "210"
            }]
        
        # Add requirements
        if "requirements" not in layout_data:
            layout_data["requirements"] = {
                "wheelchair_user": wheelchair_mode,
                "min_clearance": 150 if wheelchair_mode else 120,
                "turning_space": "150x150" if wheelchair_mode else "120x120",
                "pathway_width": 150 if wheelchair_mode else 120
            }
        
        return layout_data
    
    def optimize_layout(self, 
                       layout_data: Union[Dict, str], 
                       wheelchair_mode: bool = True,
                       temperature: float = 0.7,
                       max_new_tokens: int = 1024) -> Dict:
        """
        Optimize a room layout using the trained model
        
        Args:
            layout_data: Dictionary with room layout or JSON string
            wheelchair_mode: Whether to optimize for wheelchair accessibility
            temperature: Generation temperature (0.0-1.0)
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Optimized layout dictionary
        """
        if not self.model:
            if not self.load_model():
                return {"error": "Failed to load model"}
        
        # Parse JSON string if needed
        if isinstance(layout_data, str):
            try:
                layout_data = json.loads(layout_data)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON: {e}"}
        
        # Validate input
        is_valid, error_msg = self.validate_input(layout_data)
        if not is_valid:
            return {"error": error_msg}
        
        # Prepare input
        layout_data = self.prepare_input(layout_data, wheelchair_mode)
        
        # Generate instruction
        if wheelchair_mode:
            instruction = "Optimize this room layout for wheelchair accessibility following DIN 18040-2 standards. Ensure 150cm clearances and turning spaces."
        else:
            instruction = "Optimize this room layout following DIN 18040-2 standards. Ensure proper clearances and circulation paths."
        
        # Format prompt
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{json.dumps(layout_data, separators=(',', ':'))}

### Response:
"""
        
        print(f"\nOptimizing layout for {'wheelchair' if wheelchair_mode else 'standard'} accessibility...")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        # Try to parse the JSON output
        try:
            # Find the first { and last } to extract JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                optimized_layout = json.loads(json_str)
                
                print("✅ Layout optimized successfully!")
                return optimized_layout
            else:
                return {"error": "No valid JSON in model output", "raw_output": response}
                
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse model output: {e}", "raw_output": response}
    
    def optimize_from_file(self, input_file: str, output_file: str = None, wheelchair_mode: bool = True):
        """
        Optimize a layout from a JSON file
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to save optimized layout (optional)
            wheelchair_mode: Whether to optimize for wheelchair accessibility
        """
        print(f"\nReading layout from: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return None
        
        # Optimize
        optimized = self.optimize_layout(layout_data, wheelchair_mode)
        
        # Handle errors
        if "error" in optimized:
            print(f"❌ Optimization failed: {optimized['error']}")
            if "raw_output" in optimized:
                print(f"Raw output: {optimized['raw_output'][:200]}...")
            return None
        
        # Save if output file specified
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(optimized, f, indent=2)
                print(f"✅ Optimized layout saved to: {output_file}")
            except Exception as e:
                print(f"⚠️ Error saving file: {e}")
        
        return optimized
    
    def batch_optimize(self, layouts: List[Dict], wheelchair_mode: bool = True) -> List[Dict]:
        """
        Optimize multiple layouts in batch
        
        Args:
            layouts: List of layout dictionaries
            wheelchair_mode: Whether to optimize for wheelchair accessibility
        
        Returns:
            List of optimized layouts
        """
        results = []
        total = len(layouts)
        
        print(f"\nProcessing {total} layouts...")
        
        for i, layout in enumerate(layouts, 1):
            print(f"\n[{i}/{total}] Processing layout...")
            optimized = self.optimize_layout(layout, wheelchair_mode)
            results.append(optimized)
            
            # Clear memory periodically
            if i % 10 == 0 and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
        
        print(f"\n✅ Batch processing complete: {total} layouts processed")
        return results


# Example usage functions
def example_basic():
    """Basic usage example"""
    optimizer = DINLayoutOptimizer()
    
    # Example layout
    layout = {
        "room": {"width": 500, "height": 400},
        "furniture": [
            {"name": "Bed", "width": 180, "height": 200, "x": 20, "y": 20},
            {"name": "Wardrobe", "width": 150, "height": 60, "x": 250, "y": 50},
            {"name": "Desk", "width": 120, "height": 70, "x": 100, "y": 200}
        ]
    }
    
    # Optimize for wheelchair access
    optimized = optimizer.optimize_layout(layout, wheelchair_mode=True)
    
    if "error" not in optimized:
        print("\n" + "="*60)
        print("OPTIMIZED LAYOUT")
        print("="*60)
        print(json.dumps(optimized, indent=2))
    
    return optimized


def example_from_files():
    """Example using file input/output"""
    optimizer = DINLayoutOptimizer()
    
    # Process files from your existing layouts
    input_files = ["Input-Layouts/room-layout-3.json"]
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Skipping {input_file} - not found")
            continue
        
        output_file = input_file.replace(".json", "_optimized.json")
        print(f"\nProcessing: {input_file}")
        
        optimized = optimizer.optimize_from_file(
            input_file, 
            output_file,
            wheelchair_mode=True
        )


# Main execution
if __name__ == "__main__":
    import os
    
    print("DIN 18040-2 Layout Optimization - Inference")
    print("\nOptions:")
    print("1. Test with example layout")
    print("2. Optimize your existing room-layout.json files")
    print("3. Interactive mode (enter custom JSON)")
    
    choice = input("\nSelect option (1-3): ")
    
    if choice == "1":
        # Run basic example
        example_basic()
    
    elif choice == "2":
        # Process existing files
        example_from_files()
    
    elif choice == "3":
        # Interactive mode
        optimizer = DINLayoutOptimizer()
        
        print("\nInteractive mode - Enter layout JSON:")
        print("(Paste your JSON and press Enter twice when done)")
        
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                break
        
        json_str = "\n".join(lines)
        
        try:
            layout = json.loads(json_str)
            wheelchair = input("\nOptimize for wheelchair access? (y/n): ").lower() == 'y'
            
            optimized = optimizer.optimize_layout(layout, wheelchair_mode=wheelchair)
            
            if "error" not in optimized:
                print("\n" + "="*60)
                print("OPTIMIZED LAYOUT")
                print("="*60)
                print(json.dumps(optimized, indent=2))
                
                save = input("\nSave to file? (y/n): ").lower() == 'y'
                if save:
                    filename = input("Filename (e.g., optimized.json): ")
                    with open(filename, 'w') as f:
                        json.dump(optimized, f, indent=2)
                    print(f"✅ Saved to {filename}")
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
    
    else:
        print("Invalid option")