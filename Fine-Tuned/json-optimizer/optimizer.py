import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import re

class RoomLayoutOptimizer:
    def __init__(self, model_path="llama3.2-3b-furniture-adapted", base_model="meta-llama/Llama-3.2-3B-Instruct"):
        """Initialize the room layout optimizer with the fine-tuned model."""
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load the fine-tuned model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_4bit=True,
                device_map="auto"
            )
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(self.model, model_path)
            print("Fine-tuned model loaded successfully")
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            print("Loading base model instead...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_4bit=True,
                device_map="auto"
            )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
    
    def create_prompt_from_json(self, room_layout):
        """Convert JSON room layout to text prompt for the model."""
        room = room_layout["room"]
        furniture_list = room_layout["furniture"]
        
        # Convert room dimensions (assuming pixels to feet conversion)
        room_width_ft = round(room["width"] / 67)  # rough conversion
        room_height_ft = round(room["height"] / 67)
        
        # Extract furniture information
        furniture_names = [f["name"] for f in furniture_list]
        furniture_str = ", ".join(furniture_names)
        
        # Create prompt similar to training data
        prompt = f"""Optimize furniture arrangement in a bedroom for elderly comfort and accessibility.
Room size: {room_width_ft}x{room_height_ft} ft. Furniture: {furniture_str}. Constraints: 110 cm clear paths, wheelchair accessibility.

Input JSON:
{json.dumps(room_layout, indent=2)}

Provide optimized furniture arrangement as JSON with same structure but improved positions:"""
        
        return prompt
    
    def extract_json_from_response(self, response_text):
        """Extract JSON from model response."""
        # Remove the input prompt from response
        if "Provide optimized furniture arrangement as JSON" in response_text:
            response_text = response_text.split("Provide optimized furniture arrangement as JSON")[1]
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                # Clean up common issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return None
        return None
    
    def apply_optimization_rules(self, input_layout):
        """Apply basic optimization rules to improve the layout."""
        optimized_layout = json.deepcopy(input_layout)
        room_width = input_layout["room"]["width"]
        room_height = input_layout["room"]["height"]
        
        # Clear path requirement (110cm ≈ 74 pixels)
        min_clearance = 74
        
        for i, furniture in enumerate(optimized_layout["furniture"]):
            name = furniture["name"].lower()
            
            # Basic positioning rules based on furniture type
            if "bed" in name:
                # Place bed along a wall with clearance
                furniture["x"] = min_clearance
                furniture["y"] = min_clearance
            
            elif "wardrobe" in name or "closet" in name:
                # Place wardrobe along a wall opposite to bed
                furniture["x"] = room_width - furniture["width"] - min_clearance
                furniture["y"] = min_clearance
            
            elif "chair" in name:
                # Place chair with good access
                furniture["x"] = min_clearance
                furniture["y"] = room_height - furniture["height"] - min_clearance
            
            elif "table" in name or "desk" in name:
                # Place table/desk along wall with chair access
                furniture["x"] = room_width // 2
                furniture["y"] = min_clearance
            
            elif "sofa" in name:
                # Place sofa with good room access
                furniture["x"] = (room_width - furniture["width"]) // 2
                furniture["y"] = room_height // 2
        
        return optimized_layout
    
    def optimize_layout(self, room_layout_json, use_model=True):
        """
        Optimize room layout using either the fine-tuned model or rule-based approach.
        
        Args:
            room_layout_json: Input room layout as JSON
            use_model: Whether to use the fine-tuned model or rule-based optimization
        
        Returns:
            Optimized room layout as JSON
        """
        if use_model:
            try:
                # Create prompt from JSON
                prompt = self.create_prompt_from_json(room_layout_json)
                
                # Generate response
                response = self.pipe(
                    prompt,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )[0]["generated_text"]
                
                print("Raw model response:")
                print(response)
                print("\n" + "="*50 + "\n")
                
                # Try to extract JSON from response
                optimized_json = self.extract_json_from_response(response)
                
                if optimized_json:
                    return optimized_json
                else:
                    print("Failed to extract valid JSON from model response. Using rule-based optimization.")
                    return self.apply_optimization_rules(room_layout_json)
                    
            except Exception as e:
                print(f"Error during model inference: {e}")
                print("Falling back to rule-based optimization.")
                return self.apply_optimization_rules(room_layout_json)
        else:
            return self.apply_optimization_rules(room_layout_json)

def create_json_training_data():
    """
    Create training data in JSON format for better model training.
    This function shows how to restructure your training data.
    """
    
    # Example of how to convert your text-based training data to JSON format
    training_examples = [
        {
            "input_layout": {
                "room": {"width": 800, "height": 600},
                "furniture": [
                    {"name": "King Bed", "x": 100, "y": 100, "width": 180, "height": 180},
                    {"name": "Wardrobe", "x": 500, "y": 300, "width": 60, "height": 60},
                    {"name": "Chair", "x": 300, "y": 400, "width": 60, "height": 60}
                ]
            },
            "optimized_layout": {
                "room": {"width": 800, "height": 600},
                "furniture": [
                    {"name": "King Bed", "x": 74, "y": 74, "width": 180, "height": 180},
                    {"name": "Wardrobe", "x": 666, "y": 74, "width": 60, "height": 60},
                    {"name": "Chair", "x": 74, "y": 466, "width": 60, "height": 60}
                ]
            },
            "constraints": "110 cm clear paths, wheelchair accessibility"
        }
    ]
    
    # Convert to JSONL format for training
    jsonl_data = []
    for example in training_examples:
        prompt = f"""Optimize furniture arrangement for elderly accessibility.
Constraints: {example['constraints']}

Input layout:
{json.dumps(example['input_layout'], indent=2)}

Optimized layout:"""
        
        completion = json.dumps(example['optimized_layout'], indent=2)
        
        jsonl_data.append({
            "prompt": prompt,
            "completion": completion
        })
    
    return jsonl_data

# Example usage
def main():
    # Load the room layout from your JSON file
    with open('Input-Layouts/room-layout-2.json', 'r') as f:
        room_layout = json.load(f)
    
    print("Original Layout:")
    print(json.dumps(room_layout, indent=2))
    print("\n" + "="*50 + "\n")
    
    # Initialize the optimizer
    optimizer = RoomLayoutOptimizer()
    
    # Optimize the layout
    optimized_layout = optimizer.optimize_layout(room_layout, use_model=True)
    
    print("Optimized Layout:")
    print(json.dumps(optimized_layout, indent=2))
    
    # Save the optimized layout
    with open('optimized-room-layout.json', 'w') as f:
        json.dump(optimized_layout, f, indent=2)
    
    print("\nOptimized layout saved to 'optimized-room-layout.json'")

if __name__ == "__main__":
    main()