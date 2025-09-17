import json
import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import re

class RoomLayoutOptimizer:
    def __init__(self, model_path="llama3.2-3b-json-furniture", base_model="meta-llama/Llama-3.2-3B-Instruct"):
        """Initialize the room layout optimizer with the fine-tuned model."""
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load the fine-tuned model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            device_map="auto"
        )
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(self.model, model_path)
        print("Fine-tuned model loaded successfully")
        
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
        
        # Create a more concise prompt to leave room for the full response
        prompt = f"""Optimize furniture arrangement in a bedroom for elderly comfort and accessibility.
Room: {room_width_ft}x{room_height_ft} ft. Furniture: {furniture_str}. Constraints: 110 cm clear paths, wheelchair accessibility.

Input:
{json.dumps(room_layout, indent=2)}

Optimized layout:"""
        
        return prompt
    
    def extract_json_from_response(self, response_text):
        """Extract and repair JSON from model response."""
        print(f"Response text length: {len(response_text)}")
        print(f"Full response text:\n{response_text}")
        
        # The response should be clean JSON, try parsing it directly first
        response_text = response_text.strip()
        
        # Remove any leading/trailing whitespace and potential markdown
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # First attempt: try parsing the response directly
        try:
            parsed_json = json.loads(response_text)
            print("Successfully parsed JSON directly!")
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"Direct JSON parsing failed: {e}")
        
        # Second attempt: simple extraction
        simple_result = self.simple_json_extract(response_text)
        if simple_result:
            print("Successfully parsed with simple extraction!")
            return simple_result
        
        # Remove the input prompt if it was echoed back
        if "Optimized layout:" in response_text:
            response_text = response_text.split("Optimized layout:")[-1].strip()

        # Try to find JSON in the response - look for complete JSON blocks
        json_patterns = [
            r'\{(?:[^{}]|{[^{}]*})*\}',  # Balanced braces
            r'\{.*?\}(?=\s*$)',  # Complete JSON at end
            r'\{.*?\}(?=\s*\n\s*$)',  # Complete JSON with newline
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON structure
            r'\{.*\}',  # Any JSON-like structure
        ]
        
        json_str = None
        for pattern in json_patterns:
            json_match = re.search(pattern, response_text, re.DOTALL | re.MULTILINE)
            if json_match:
                json_str = json_match.group(0)
                print(f"Found JSON with pattern: {pattern}")
                break
        
        if not json_str:
            print("No JSON pattern found in response")
            return None

        print(f"Extracted JSON length: {len(json_str)}")
        print(f"JSON string: {json_str}")

        # --- Enhanced JSON Repair ---
        original_json = json_str
        
        # Remove trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        # Remove double commas
        json_str = re.sub(r',\s*,', ',', json_str)
        # Add missing commas between objects
        json_str = re.sub(r'}\s*{', '},{', json_str)
        # Remove non-printable characters
        json_str = re.sub(r'[^\x20-\x7E\n\r\t]', '', json_str)
        
        # Try to handle incomplete JSON by checking bracket balance
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        print(f"Brace balance: {open_braces} open, {close_braces} close")
        
        # Add missing closing brackets/braces
        missing_braces = max(0, open_braces - close_braces)
        missing_brackets = max(0, open_brackets - close_brackets)
        
        if missing_braces > 0 or missing_brackets > 0:
            print(f"Adding {missing_braces} closing braces and {missing_brackets} closing brackets")
            json_str += '}' * missing_braces
            json_str += ']' * missing_brackets
        
        # Try to parse the repaired JSON
        try:
            parsed_json = json.loads(json_str)
            print("Successfully parsed JSON after repair")
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON decode error after repair: {e}")
            print(f"Final JSON string: {json_str}")
            
            # Last resort: try to extract partial JSON and reconstruct
            return self.reconstruct_partial_json(original_json, response_text)

    def simple_json_extract(self, response_text):
        """Simple JSON extraction method as backup."""
        # Clean the response
        response_text = response_text.strip()
        
        # Find the first '{' and last '}'
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return None
        
        json_candidate = response_text[start_idx:end_idx + 1]
        
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            return None
        """Attempt to reconstruct incomplete JSON from partial response."""
        try:
            # Try to find the furniture array at least
            furniture_match = re.search(r'"furniture":\s*\[(.*?)\]', json_str, re.DOTALL)
            if furniture_match:
                print("Attempting to reconstruct from partial furniture data...")
                
                # Build a minimal valid structure
                reconstructed = {
                    "room": {"width": 800, "height": 600},  # Default values
                    "furniture": [],
                    "openings": []  # Will be empty if not found
                }
                
                # Try to parse individual furniture objects
                furniture_text = furniture_match.group(1)
                furniture_objects = re.findall(r'\{[^}]+\}', furniture_text)
                
                for obj_str in furniture_objects:
                    try:
                        # Fix common issues in furniture objects
                        obj_str = re.sub(r',\s*}', '}', obj_str)
                        furniture_obj = json.loads(obj_str)
                        reconstructed["furniture"].append(furniture_obj)
                    except json.JSONDecodeError:
                        continue
                
                if reconstructed["furniture"]:
                    print(f"Reconstructed {len(reconstructed['furniture'])} furniture objects")
                    return reconstructed
                    
        except Exception as e:
            print(f"Failed to reconstruct partial JSON: {e}")
        
        return None
    
    def optimize_layout(self, room_layout_json, max_retries=3):
        """
        Optimize room layout using the fine-tuned model only.
        
        Args:
            room_layout_json: Input room layout as JSON
            max_retries: Number of retries if JSON parsing fails
        
        Returns:
            Optimized room layout as JSON
        """
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}")
                
                # Create prompt from JSON
                prompt = self.create_prompt_from_json(room_layout_json)
                
                # Generate response with increased token limit and adjusted parameters
                response = self.pipe(
                    prompt,
                    max_new_tokens=1200,  # Much higher limit for complete JSON
                    do_sample=True,
                    temperature=0.3,  # Lower temperature for more consistent JSON
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False,  # Only return generated text, not input
                    truncation=False  # Don't truncate the response
                )[0]["generated_text"]
                
                print("Raw model response:")
                print(f"Response length: {len(response)} characters")
                print(response)
                print("\n" + "="*50 + "\n")
                
                # DEBUG: Let's also try parsing the raw response directly
                try:
                    if response.strip().startswith('{'):
                        direct_parse = json.loads(response.strip())
                        print("SUCCESS: Raw response is valid JSON!")
                        if self.validate_json_structure(direct_parse, room_layout_json):
                            print("SUCCESS: Raw response structure is valid!")
                            return direct_parse
                        else:
                            print("Raw response structure is invalid")
                except json.JSONDecodeError as e:
                    print(f"Raw response is not valid JSON: {e}")
                except Exception as e:
                    print(f"Error parsing raw response: {e}")
                
                # Try to extract JSON from response
                optimized_json = self.extract_json_from_response(response)
                
                if optimized_json and self.validate_json_structure(optimized_json, room_layout_json):
                    print("Successfully generated valid optimized layout!")
                    return optimized_json
                else:
                    print(f"Invalid JSON structure on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        print("Retrying with different parameters...")
                    
            except Exception as e:
                print(f"Error during model inference (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    print("Retrying...")
                
        # If all retries failed, raise an exception
        raise RuntimeError(f"Failed to generate valid optimized layout after {max_retries} attempts")
    
    def validate_json_structure(self, optimized_json, original_json):
        """Validate that the optimized JSON has the correct structure."""
        try:
            # Check required keys
            if not all(key in optimized_json for key in ["room", "furniture"]):
                print("Missing required keys in optimized JSON")
                return False
            
            # Check room structure
            room = optimized_json["room"]
            if not all(key in room for key in ["width", "height"]):
                print("Invalid room structure")
                return False
            
            # Check furniture structure
            furniture_list = optimized_json["furniture"]
            if not isinstance(furniture_list, list):
                print("Furniture is not a list")
                return False
            
            # Validate each furniture object
            required_furniture_keys = ["name", "x", "y", "width", "height"]
            for i, furniture in enumerate(furniture_list):
                if not all(key in furniture for key in required_furniture_keys):
                    print(f"Invalid furniture object at index {i}")
                    return False
                
                # Check that coordinates are within room bounds
                if (furniture["x"] < 0 or furniture["y"] < 0 or 
                    furniture["x"] + furniture["width"] > room["width"] or
                    furniture["y"] + furniture["height"] > room["height"]):
                    print(f"Furniture {furniture['name']} is outside room bounds")
                    # Don't return False here, just warn - the model might have good reasons
            
            print("JSON structure validation passed")
            return True
            
        except Exception as e:
            print(f"Error during JSON validation: {e}")
            return False

# Example usage
def main():
    # Load the room layout from your JSON file
    with open('Input-Layouts/room-layout-chairs-table.json', 'r') as f:
        room_layout = json.load(f)
    print("\nInput Layout Loaded")
    print("=" * 50)
    
    try:
        # Initialize the optimizer
        optimizer = RoomLayoutOptimizer()
        
        # Optimize the layout using only the model
        optimized_layout = optimizer.optimize_layout(room_layout)
        
        # Save the optimized layout
        with open('optimized-room-layout.json', 'w') as f:
            json.dump(optimized_layout, f, indent=2)
        
        print("\nOptimized layout saved to 'optimized-room-layout.json'")
        
    except RuntimeError as e:
        print(f"\nFailed to optimize layout: {e}")
        print("The fine-tuned model may need additional training or parameter adjustments.")

if __name__ == "__main__":
    main()