import json
from google import genai
from typing import Dict, List, Tuple, Any
import math

class RoomLayoutOptimizer:
    def __init__(self, api_key: str):
        """
        Initialize the Room Layout Optimizer with Google AI Studio API key
        
        Args:
            api_key (str): Your Google AI Studio API key
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"
        
    def load_constraints(self, constraints_path: str) -> Dict:
        """Load barrier-free constraints from JSON file"""
        with open(constraints_path, 'r') as f:
            return json.load(f)
    
    def load_layout(self, layout_path: str) -> Dict:
        """Load room layout from JSON file"""
        with open(layout_path, 'r') as f:
            return json.load(f)
    
    def validate_layout(self, layout: Dict, constraints: Dict) -> List[str]:
        """
        Validate current layout against constraints and return violations
        
        Args:
            layout (Dict): Current room layout
            constraints (Dict): Barrier-free constraints
            
        Returns:
            List[str]: List of constraint violations
        """
        violations = []
        room_width, room_height = layout['room']['width'], layout['room']['height']
        furniture = layout['furniture']
        
        # Check minimum path width between furniture
        for i, item1 in enumerate(furniture):
            for j, item2 in enumerate(furniture[i+1:], i+1):
                distance = self._calculate_minimum_distance(item1, item2)
                if distance < constraints['room_constraints']['min_path_width']:
                    violations.append(f"Insufficient path width ({distance}cm) between {item1['name']} and {item2['name']}. Required: {constraints['room_constraints']['min_path_width']}cm")
        
        # Check furniture-specific clearances
        for item in furniture:
            furniture_type = item['name']
            if furniture_type in constraints['furniture_specific_clearances']:
                clearance_reqs = constraints['furniture_specific_clearances'][furniture_type]
                violations.extend(self._check_furniture_clearances(item, clearance_reqs, furniture, room_width, room_height))
        
        return violations
    
    def _calculate_minimum_distance(self, item1: Dict, item2: Dict) -> float:
        """Calculate minimum distance between two furniture items"""
        # Get boundaries of both items
        x1_min, y1_min = item1['x'], item1['y']
        x1_max, y1_max = x1_min + item1['width'], y1_min + item1['height']
        
        x2_min, y2_min = item2['x'], item2['y']
        x2_max, y2_max = x2_min + item2['width'], y2_min + item2['height']
        
        # Calculate distance between rectangles
        dx = max(0, max(x1_min - x2_max, x2_min - x1_max))
        dy = max(0, max(y1_min - y2_max, y2_min - y1_max))
        
        return max(dx, dy)  # Return the minimum clearance distance
    
    def _check_furniture_clearances(self, item: Dict, clearance_reqs: Dict, all_furniture: List[Dict], room_width: int, room_height: int) -> List[str]:
        """Check if furniture item meets clearance requirements"""
        violations = []
        
        # Check front clearance if required
        if 'front_clearance' in clearance_reqs:
            required_clearance = clearance_reqs['front_clearance']
            # Check if there's enough space in front of the item
            front_space = self._calculate_front_space(item, all_furniture, room_width, room_height)
            if front_space < required_clearance:
                violations.append(f"{item['name']} has insufficient front clearance ({front_space}cm). Required: {required_clearance}cm")
        
        return violations
    
    def _calculate_front_space(self, item: Dict, all_furniture: List[Dict], room_width: int, room_height: int) -> float:
        """Calculate available space in front of furniture item"""
        # Simplified calculation - assumes front is towards positive Y direction
        item_front_y = item['y'] + item['height']
        space_to_wall = room_height - item_front_y
        
        # Check for other furniture blocking the front
        min_space = space_to_wall
        for other_item in all_furniture:
            if other_item == item:
                continue
            
            # Check if other item is in front and overlaps horizontally
            if (other_item['y'] >= item_front_y and 
                other_item['x'] < item['x'] + item['width'] and 
                other_item['x'] + other_item['width'] > item['x']):
                
                space_to_furniture = other_item['y'] - item_front_y
                min_space = min(min_space, space_to_furniture)
        
        return min_space
    
    def create_optimization_prompt(self, layout: Dict, constraints: Dict, violations: List[str]) -> str:
        """
        Create a detailed prompt for the AI model to optimize the layout
        
        Args:
            layout (Dict): Current room layout
            constraints (Dict): Barrier-free constraints
            violations (List[str]): Current constraint violations
            
        Returns:
            str: Formatted prompt for the AI model
        """
        prompt = f"""
You are an expert in barrier-free (accessible) interior design. Optimize this room layout to fix accessibility violations.

CURRENT ROOM LAYOUT:
{json.dumps(layout, indent=2)}

ACCESSIBILITY VIOLATIONS TO FIX:
{chr(10).join(f"- {violation}" for violation in violations)}

CONSTRAINTS TO FOLLOW:
- Minimum path width: {constraints['room_constraints']['min_path_width']}cm between all furniture
- Furniture clearances as specified in constraints
- Room boundaries: {layout['room']['width']}cm x {layout['room']['height']}cm
- All furniture must stay within room bounds

INSTRUCTIONS:
1. Move furniture to resolve ALL violations listed above
2. Maintain minimum {constraints['room_constraints']['min_path_width']}cm clearance between items
3. Keep all furniture within room boundaries
4. Use only rotation values: 0, 90, 180, 270
5. Return ONLY valid JSON - no explanations or markdown

REQUIRED OUTPUT FORMAT - RETURN EXACTLY THIS STRUCTURE:
{{
  "room": {{
    "width": {layout['room']['width']},
    "height": {layout['room']['height']}
  }},
  "furniture": [
    {{
      "name": "Sofa",
      "x": 100,
      "y": 100,
      "width": 200,
      "height": 90,
      "zHeight": "85",
      "rotation": 0
    }}
  ],
  "openings": []
}}

Return the optimized layout JSON now:"""
        return prompt
    
    def optimize_layout(self, layout: Dict, constraints: Dict, max_iterations: int = 5) -> Dict:
        """
        Optimize room layout using AI model
        
        Args:
            layout (Dict): Current room layout
            constraints (Dict): Barrier-free constraints
            max_iterations (int): Maximum optimization iterations
            
        Returns:
            Dict: Optimized room layout
        """
        current_layout = layout.copy()
        
        for iteration in range(max_iterations):
            print(f"Optimization iteration {iteration + 1}/{max_iterations}")
            
            # Validate current layout
            violations = self.validate_layout(current_layout, constraints)
            
            if not violations:
                print("Layout is already optimized!")
                break
            
            print(f"Found {len(violations)} violations:")
            for violation in violations[:5]:  # Show first 5 violations
                print(f"  - {violation}")
            
            # Create optimization prompt
            prompt = self.create_optimization_prompt(current_layout, constraints, violations)
            
            try:
                # Get AI response
                print("Sending request to AI...")
                response = self.client.models.generate_content(
                    model=self.model_name, 
                    contents=prompt
                )
                
                print("Received AI response, processing...")
                # Extract JSON from response
                response_text = response.text.strip()
                print(f"Response length: {len(response_text)} characters")
                
                # Try to find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    print("Found JSON in response, parsing...")
                    
                    try:
                        optimized_layout = json.loads(json_text)
                        print("Successfully parsed JSON layout")
                        
                        # Validate the optimized layout
                        new_violations = self.validate_layout(optimized_layout, constraints)
                        
                        if len(new_violations) < len(violations):
                            print(f"Improved! Violations reduced from {len(violations)} to {len(new_violations)}")
                            current_layout = optimized_layout
                        elif len(new_violations) == 0:
                            print("Perfect! All violations resolved!")
                            current_layout = optimized_layout
                        else:
                            print(f"No improvement. Still {len(new_violations)} violations")
                            if len(new_violations) <= len(violations):
                                current_layout = optimized_layout  # Accept equal or better
                            
                    except json.JSONDecodeError as json_err:
                        print(f"JSON parsing error: {json_err}")
                        print("Raw JSON text:")
                        print(json_text[:500] + "..." if len(json_text) > 500 else json_text)
                        break
                        
                else:
                    print("Could not find valid JSON brackets in AI response")
                    print("Response preview:")
                    print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
                    break
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print("This might be due to the AI response format. Continuing...")
                break
            except Exception as e:
                print(f"Error during optimization: {e}")
                print("Continuing with current layout...")
                break
        
        # Final status
        final_violations = self.validate_layout(current_layout, constraints)
        if len(final_violations) == 0:
            print("\nSUCCESS: All accessibility violations have been resolved!")
        else:
            print(f"\nPARTIAL SUCCESS: Reduced violations from {len(self.validate_layout(layout, constraints))} to {len(final_violations)}")
            print("Remaining violations:")
            for violation in final_violations[:3]:
                print(f"  - {violation}")
        
        return current_layout
    
    def save_layout(self, layout: Dict, output_path: str):
        """Save optimized layout to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(layout, f, indent=2)
        print(f"Optimized layout saved to {output_path}")
    
    def generate_optimization_report(self, original_layout: Dict, optimized_layout: Dict, constraints: Dict) -> Dict:
        """
        Generate a report comparing original and optimized layouts
        
        Args:
            original_layout (Dict): Original room layout
            optimized_layout (Dict): Optimized room layout
            constraints (Dict): Barrier-free constraints
            
        Returns:
            Dict: Optimization report
        """
        original_violations = self.validate_layout(original_layout, constraints)
        optimized_violations = self.validate_layout(optimized_layout, constraints)
        
        # Calculate furniture movements
        movements = []
        for i, orig_item in enumerate(original_layout['furniture']):
            opt_item = optimized_layout['furniture'][i]
            distance = math.sqrt((opt_item['x'] - orig_item['x'])**2 + (opt_item['y'] - orig_item['y'])**2)
            if distance > 0:
                movements.append({
                    'furniture': orig_item['name'],
                    'distance_moved': round(distance, 2),
                    'rotation_changed': orig_item['rotation'] != opt_item['rotation']
                })
        
        report = {
            'optimization_summary': {
                'violations_before': len(original_violations),
                'violations_after': len(optimized_violations),
                'improvement': len(original_violations) - len(optimized_violations),
                'optimization_successful': len(optimized_violations) == 0
            },
            'remaining_violations': optimized_violations,
            'furniture_movements': movements,
            'total_items_moved': len(movements)
        }
        
        return report


from config import Config

def main():
    API_KEY = Config.OPENAI_API_KEY
    if not API_KEY:
        raise ValueError("API_KEY environment variable not set. Please set it before running.")
    optimizer = RoomLayoutOptimizer(API_KEY)
    
    # Specify file paths here
    constraints_path = r"constraints/bedroom_barrier_free_constraints_consolidated.json"
    layout_path = r"Input-Layouts/room-layout-1.json"
    output_path = r"Outputs/FT/optimized_room_layout.json"
    report_path = r"Outputs/FT/optimization_report.json"
    
    # Load constraints and layout
    constraints = optimizer.load_constraints(constraints_path)
    layout = optimizer.load_layout(layout_path) 
    
    print("Starting room layout optimization...")
    print(f"Room dimensions: {layout['room']['width']}cm x {layout['room']['height']}cm")
    print(f"Number of furniture items: {len(layout['furniture'])}")

    optimized_layout = optimizer.optimize_layout(layout, constraints)
    
    report = optimizer.generate_optimization_report(layout, optimized_layout, constraints)
    
    optimizer.save_layout(optimized_layout, output_path)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Violations before: {report['optimization_summary']['violations_before']}")
    print(f"Violations after: {report['optimization_summary']['violations_after']}")
    print(f"Items moved: {report['total_items_moved']}")
    print(f"Optimization successful: {report['optimization_summary']['optimization_successful']}")
    
    if report['remaining_violations']:
        print("\nRemaining violations:")
        for violation in report['remaining_violations']:
            print(f"  - {violation}")

if __name__ == "__main__":
    main()