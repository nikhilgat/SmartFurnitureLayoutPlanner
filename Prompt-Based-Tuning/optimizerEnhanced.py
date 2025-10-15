import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import google.generativeai as genai
from typing import Dict, List
import math
import copy

class RoomLayoutOptimizer:
    def __init__(self, api_key: str):
        """
        Initialize the Room Layout Optimizer with Google AI Studio API key
        
        Args:
            api_key (str): Your Google AI Studio API key
        """
        self.client = genai.configure(api_key=api_key)
        self.model_name = "gemini-2.0-flash-exp"
        
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
                min_required = constraints['room_constraints']['min_path_width']
                if distance < min_required:
                    violations.append(f"Insufficient path width ({distance:.0f}cm) between {item1['name']} and {item2['name']}. Required: {min_required}cm")
        
        # Check furniture-specific clearances
        for item in furniture:
            furniture_type = item['name']
            if furniture_type in constraints['furniture_specific_clearances']:
                clearance_reqs = constraints['furniture_specific_clearances'][furniture_type]
                violations.extend(self._check_furniture_clearances(item, clearance_reqs, furniture, room_width, room_height))
        
        # Check bounds violations
        for item in furniture:
            if (item['x'] < 0 or item['y'] < 0 or 
                item['x'] + item['width'] > room_width or 
                item['y'] + item['height'] > room_height):
                violations.append(f"{item['name']} is outside room boundaries")
        
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
        
        return min(dx, dy) if dx > 0 and dy > 0 else max(dx, dy)
    
    def _check_furniture_clearances(self, item: Dict, clearance_reqs: Dict, all_furniture: List[Dict], room_width: int, room_height: int) -> List[str]:
        """Check if furniture item meets clearance requirements"""
        violations = []
        
        # Check front clearance if required
        if 'front_clearance' in clearance_reqs:
            required_clearance = clearance_reqs['front_clearance']
            front_space = self._calculate_front_space(item, all_furniture, room_width, room_height)
            if front_space < required_clearance:
                violations.append(f"{item['name']} has insufficient front clearance ({front_space:.0f}cm). Required: {required_clearance}cm")
        
        return violations
    
    def _calculate_front_space(self, item: Dict, all_furniture: List[Dict], room_width: int, room_height: int) -> float:
        """Calculate available space in front of furniture item based on rotation"""
        rotation = item.get('rotation', 0)
        
        # Determine front direction based on rotation
        if rotation == 0:  # Front faces up (positive Y)
            front_edge = item['y'] + item['height']
            space_to_wall = room_height - front_edge
            check_axis = 'y'
            check_min = item['x']
            check_max = item['x'] + item['width']
        elif rotation == 90:  # Front faces right (positive X)
            front_edge = item['x'] + item['width']
            space_to_wall = room_width - front_edge
            check_axis = 'x'
            check_min = item['y']
            check_max = item['y'] + item['height']
        elif rotation == 180:  # Front faces down (negative Y)
            front_edge = item['y']
            space_to_wall = front_edge
            check_axis = 'y'
            check_min = item['x']
            check_max = item['x'] + item['width']
        else:  # rotation == 270, Front faces left (negative X)
            front_edge = item['x']
            space_to_wall = front_edge
            check_axis = 'x'
            check_min = item['y']
            check_max = item['y'] + item['height']
        
        min_space = space_to_wall
        
        # Check for other furniture blocking the front
        for other_item in all_furniture:
            if other_item == item:
                continue
            
            if check_axis == 'y':
                if rotation == 0:  # Front faces up
                    if (other_item['y'] >= front_edge and 
                        other_item['x'] < check_max and 
                        other_item['x'] + other_item['width'] > check_min):
                        space_to_furniture = other_item['y'] - front_edge
                        min_space = min(min_space, space_to_furniture)
                else:  # rotation == 180, Front faces down
                    if (other_item['y'] + other_item['height'] <= front_edge and 
                        other_item['x'] < check_max and 
                        other_item['x'] + other_item['width'] > check_min):
                        space_to_furniture = front_edge - (other_item['y'] + other_item['height'])
                        min_space = min(min_space, space_to_furniture)
            else:  # check_axis == 'x'
                if rotation == 90:  # Front faces right
                    if (other_item['x'] >= front_edge and 
                        other_item['y'] < check_max and 
                        other_item['y'] + other_item['height'] > check_min):
                        space_to_furniture = other_item['x'] - front_edge
                        min_space = min(min_space, space_to_furniture)
                else:  # rotation == 270, Front faces left
                    if (other_item['x'] + other_item['width'] <= front_edge and 
                        other_item['y'] < check_max and 
                        other_item['y'] + other_item['height'] > check_min):
                        space_to_furniture = front_edge - (other_item['x'] + other_item['width'])
                        min_space = min(min_space, space_to_furniture)
        
        return max(0, min_space)
    
    def create_hybrid_optimization_prompt(self, layout: Dict, constraints: Dict, violations: List[str], iteration: int) -> str:
        """
        Create an enhanced prompt with specific strategies for different iterations
        """
        base_prompt = f"""
You are an expert in barrier-free (accessible) interior design and spatial optimization. Your goal is to create an optimal furniture layout that meets ALL accessibility requirements.

CURRENT ROOM LAYOUT:
{json.dumps(layout, indent=2)}

CRITICAL VIOLATIONS TO RESOLVE (ITERATION {iteration}):
{chr(10).join(f"- {violation}" for violation in violations)}

ACCESSIBILITY CONSTRAINTS:
- Minimum path width: {constraints['room_constraints']['min_path_width']}cm between ALL furniture items
- Room boundaries: {layout['room']['width']}cm x {layout['room']['height']}cm
- All furniture must stay within room bounds
- Furniture-specific clearances must be maintained

OPTIMIZATION STRATEGY FOR ITERATION {iteration}:"""

        if iteration <= 2:
            strategy = """
FOCUS: Major repositioning to create space
1. Spread furniture items apart significantly
2. Place large items (Bed, Sofa, Wardrobe) against walls
3. Keep smaller items (chairs, tables) away from large items
4. Ensure at least 150cm between major furniture pieces
5. Rotate furniture if needed to optimize space usage"""

        elif iteration <= 4:
            strategy = """
FOCUS: Fine-tuning positions and clearances
1. Adjust positions incrementally (10-30cm movements)
2. Focus on front clearance violations
3. Ensure study chair can be pulled out from table
4. Check wall clearances for wardrobes and sofas
5. Optimize corner placements"""

        else:
            strategy = """
FOCUS: Final optimization and constraint satisfaction
1. Make minimal, precise adjustments
2. Prioritize the most critical violations
3. Consider alternative furniture orientations
4. Ensure no overlapping or boundary violations
5. Double-check all clearance requirements"""

        prompt = base_prompt + strategy + f"""

CRITICAL RULES:
1. NO furniture can overlap (0cm distance = VIOLATION)
2. Study Chair must be at least 120cm from Study Table when pulled out
3. All furniture must have their required front clearances
4. Use only rotation values: 0, 90, 180, 270
5. Return ONLY valid JSON - no explanations

REQUIRED OUTPUT - RETURN EXACTLY THIS STRUCTURE:
{{
  "room": {{
    "width": {layout['room']['width']},
    "height": {layout['room']['height']}
  }},
  "furniture": [
    {{
      "name": "furniture_name",
      "x": position_x,
      "y": position_y,
      "width": furniture_width,
      "height": furniture_height,
      "zHeight": "furniture_z_height",
      "rotation": rotation_angle
    }}
  ],
  "openings": []
}}

Return the optimized layout JSON:"""
        return prompt

    def apply_heuristic_improvements(self, layout: Dict, constraints: Dict) -> Dict:
        """
        Apply heuristic-based improvements to the layout
        """
        improved_layout = copy.deepcopy(layout)
        furniture = improved_layout['furniture']
        room_width, room_height = layout['room']['width'], layout['room']['height']
        min_clearance = constraints['room_constraints']['min_path_width']
        
        # Sort furniture by size (largest first) for better placement
        furniture.sort(key=lambda x: x['width'] * x['height'], reverse=True)
        
        # Apply improvements
        for i, item in enumerate(furniture):
            # Try to place large items against walls
            if item['width'] * item['height'] > 20000:  # Large furniture
                # Try wall positions
                wall_positions = [
                    (0, item['y']),  # Left wall
                    (room_width - item['width'], item['y']),  # Right wall
                    (item['x'], 0),  # Bottom wall
                    (item['x'], room_height - item['height'])  # Top wall
                ]
                
                for new_x, new_y in wall_positions:
                    if (0 <= new_x <= room_width - item['width'] and 
                        0 <= new_y <= room_height - item['height']):
                        
                        # Check if this position reduces violations
                        test_item = copy.deepcopy(item)
                        test_item['x'], test_item['y'] = new_x, new_y
                        
                        conflicts = self._count_conflicts(test_item, furniture, constraints)
                        current_conflicts = self._count_conflicts(item, furniture, constraints)
                        
                        if conflicts < current_conflicts:
                            item['x'], item['y'] = new_x, new_y
                            break
            
            # Ensure minimum clearance from other furniture
            for j, other_item in enumerate(furniture):
                if i != j:
                    distance = self._calculate_minimum_distance(item, other_item)
                    if distance < min_clearance:
                        # Try to move current item away
                        self._push_apart(item, other_item, min_clearance, room_width, room_height)
        
        return improved_layout
    
    def _count_conflicts(self, item: Dict, all_furniture: List[Dict], constraints: Dict) -> int:
        """Count how many constraints this item violates"""
        conflicts = 0
        min_clearance = constraints['room_constraints']['min_path_width']
        
        for other_item in all_furniture:
            if item == other_item:
                continue
            distance = self._calculate_minimum_distance(item, other_item)
            if distance < min_clearance:
                conflicts += 1
        
        return conflicts
    
    def _push_apart(self, item1: Dict, item2: Dict, min_distance: float, room_width: int, room_height: int):
        """Try to push two items apart to maintain minimum distance"""
        # Calculate current centers
        c1_x = item1['x'] + item1['width'] / 2
        c1_y = item1['y'] + item1['height'] / 2
        c2_x = item2['x'] + item2['width'] / 2
        c2_y = item2['y'] + item2['height'] / 2
        
        # Calculate direction vector
        dx = c1_x - c2_x
        dy = c1_y - c2_y
        
        if dx == 0 and dy == 0:
            dx, dy = 1, 0  # Default direction if items are at same center
        
        # Normalize direction
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            dx /= length
            dy /= length
        
        # Calculate required movement
        current_distance = self._calculate_minimum_distance(item1, item2)
        required_movement = (min_distance - current_distance) / 2 + 10  # Add buffer
        
        # Try moving item1
        new_x = item1['x'] + dx * required_movement
        new_y = item1['y'] + dy * required_movement
        
        # Check bounds
        if (0 <= new_x <= room_width - item1['width'] and 
            0 <= new_y <= room_height - item1['height']):
            item1['x'] = max(0, min(new_x, room_width - item1['width']))
            item1['y'] = max(0, min(new_y, room_height - item1['height']))

    def optimize_layout(self, layout: Dict, constraints: Dict, max_iterations: int = 6) -> Dict:
        """
        Enhanced optimize layout with hybrid approach
        """
        current_layout = copy.deepcopy(layout)
        best_layout = copy.deepcopy(layout)
        best_violation_count = len(self.validate_layout(layout, constraints))
        
        print(f"Starting optimization with {best_violation_count} initial violations")
        
        for iteration in range(max_iterations):
            print(f"\nOptimization iteration {iteration + 1}/{max_iterations}")
            
            # Apply heuristic improvements every few iterations
            if iteration % 2 == 0:
                print("Applying heuristic improvements...")
                current_layout = self.apply_heuristic_improvements(current_layout, constraints)
            
            violations = self.validate_layout(current_layout, constraints)
            
            if not violations:
                print("🎉 Perfect! Layout is fully optimized!")
                return current_layout
            
            print(f"Found {len(violations)} violations:")
            for violation in violations[:3]:
                print(f"  - {violation}")
            
            # Create enhanced prompt
            prompt = self.create_hybrid_optimization_prompt(current_layout, constraints, violations, iteration + 1)
            
            try:
                print("Requesting AI optimization...")
                response = self.client.models.generate_content(
                    model=self.model_name, 
                    contents=prompt
                )
                
                response_text = response.text.strip()
                
                # Extract JSON
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    
                    try:
                        optimized_layout = json.loads(json_text)
                        new_violations = self.validate_layout(optimized_layout, constraints)
                        
                        print(f"AI response: {len(violations)} → {len(new_violations)} violations")
                        
                        # Accept if improvement or equal
                        if len(new_violations) <= len(violations):
                            current_layout = optimized_layout
                            if len(new_violations) < best_violation_count:
                                best_layout = copy.deepcopy(optimized_layout)
                                best_violation_count = len(new_violations)
                                print(f"✅ New best: {best_violation_count} violations")
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error: {e}")
                        # Continue with current layout
                        
            except Exception as e:
                print(f"Error during optimization: {e}")
                continue
        
        # Return the best layout found
        final_violations = self.validate_layout(best_layout, constraints)
        print(f"\n🏁 Optimization complete: {len(self.validate_layout(layout, constraints))} → {len(final_violations)} violations")
        
        return best_layout
    
    def visualize_layout(self, layout: Dict, constraints: Dict, output_path: str, title: str = "Room Layout"):
        """
        Create a comprehensive visualization of the room layout
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        room_width = layout['room']['width']
        room_height = layout['room']['height']
        furniture = layout['furniture']
        violations = self.validate_layout(layout, constraints)
        
        # Color scheme
        colors = {
            'Bed': '#FF6B6B',
            'Sofa': '#4ECDC4',
            'Wardrobe': '#45B7D1',
            'Study Table': '#96CEB4',
            'Study Chair': '#FFEAA7',
            'Bedside Table': '#DDA0DD',
            'Room': '#F8F9FA'
        }
        
        for ax_idx, ax in enumerate([ax1, ax2]):
            # Draw room boundaries
            room_rect = Rectangle((0, 0), room_width, room_height, 
                                linewidth=3, edgecolor='black', facecolor=colors.get('Room', '#F0F0F0'))
            ax.add_patch(room_rect)
            
            # Draw furniture
            for item in furniture:
                color = colors.get(item['name'], '#CCCCCC')
                
                # Create furniture rectangle
                furniture_rect = Rectangle(
                    (item['x'], item['y']), item['width'], item['height'],
                    linewidth=2, edgecolor='black', facecolor=color, alpha=0.8
                )
                ax.add_patch(furniture_rect)
                
                # Add furniture label
                label_x = item['x'] + item['width'] / 2
                label_y = item['y'] + item['height'] / 2
                ax.text(label_x, label_y, item['name'], 
                       ha='center', va='center', fontsize=10, fontweight='bold')
                
                # Add rotation indicator
                if item.get('rotation', 0) != 0:
                    ax.text(item['x'] + 5, item['y'] + 5, f"↻{item['rotation']}°", 
                           fontsize=8, color='red')
            
            # For the second subplot, add violation indicators
            if ax_idx == 1:
                # Draw clearance zones and violations
                min_clearance = constraints['room_constraints']['min_path_width']
                
                for i, item1 in enumerate(furniture):
                    for j, item2 in enumerate(furniture[i+1:], i+1):
                        distance = self._calculate_minimum_distance(item1, item2)
                        
                        if distance < min_clearance:
                            # Draw line between violating items
                            c1_x = item1['x'] + item1['width'] / 2
                            c1_y = item1['y'] + item1['height'] / 2
                            c2_x = item2['x'] + item2['width'] / 2
                            c2_y = item2['y'] + item2['height'] / 2
                            
                            ax.plot([c1_x, c2_x], [c1_y, c2_y], 
                                   'r--', linewidth=2, alpha=0.7)
                            
                            # Add distance label
                            mid_x = (c1_x + c2_x) / 2
                            mid_y = (c1_y + c2_y) / 2
                            ax.text(mid_x, mid_y, f'{distance:.0f}cm', 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                                   fontsize=8, ha='center', va='center', color='white')
                
                # Add clearance zones for furniture with front clearance requirements
                for item in furniture:
                    if item['name'] in constraints.get('furniture_specific_clearances', {}):
                        clearance_req = constraints['furniture_specific_clearances'][item['name']].get('front_clearance')
                        if clearance_req:
                            front_space = self._calculate_front_space(item, furniture, room_width, room_height)
                            
                            # Draw clearance zone based on rotation
                            rotation = item.get('rotation', 0)
                            
                            if rotation == 0:  # Front faces up
                                zone_rect = Rectangle(
                                    (item['x'], item['y'] + item['height']), 
                                    item['width'], min(clearance_req, room_height - item['y'] - item['height']),
                                    alpha=0.3, facecolor='yellow' if front_space >= clearance_req else 'red'
                                )
                            elif rotation == 90:  # Front faces right
                                zone_rect = Rectangle(
                                    (item['x'] + item['width'], item['y']), 
                                    min(clearance_req, room_width - item['x'] - item['width']), item['height'],
                                    alpha=0.3, facecolor='yellow' if front_space >= clearance_req else 'red'
                                )
                            elif rotation == 180:  # Front faces down
                                zone_rect = Rectangle(
                                    (item['x'], max(0, item['y'] - clearance_req)), 
                                    item['width'], min(clearance_req, item['y']),
                                    alpha=0.3, facecolor='yellow' if front_space >= clearance_req else 'red'
                                )
                            else:  # rotation == 270, Front faces left
                                zone_rect = Rectangle(
                                    (max(0, item['x'] - clearance_req), item['y']), 
                                    min(clearance_req, item['x']), item['height'],
                                    alpha=0.3, facecolor='yellow' if front_space >= clearance_req else 'red'
                                )
                            
                            ax.add_patch(zone_rect)
            
            # Set axis properties
            ax.set_xlim(-50, room_width + 50)
            ax.set_ylim(-50, room_height + 50)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Width (cm)', fontsize=12)
            ax.set_ylabel('Height (cm)', fontsize=12)
            
        # Set titles
        ax1.set_title(f'{title} - Layout View', fontsize=14, fontweight='bold')
        ax2.set_title(f'{title} - Violations & Clearances', fontsize=14, fontweight='bold')
        
        # Add violations summary
        violation_text = f"Violations: {len(violations)}"
        if violations:
            violation_text += f"\nTop Issues:\n" + "\n".join([f"• {v[:50]}..." if len(v) > 50 else f"• {v}" for v in violations[:3]])
        
        fig.text(0.02, 0.98, violation_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue" if not violations else "lightcoral"))
        
        # Add legend
        legend_elements = [patches.Patch(facecolor=colors.get(name, '#CCCCCC'), 
                                       edgecolor='black', label=name) 
                          for name in colors.keys() if name != 'Room']
        ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    def create_comparison_visualization(self, original_layout: Dict, optimized_layout: Dict, constraints: Dict, output_path: str):
        """
        Create a side-by-side comparison of original and optimized layouts
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16))
        
        layouts = [original_layout, optimized_layout]
        titles = ["Original Layout", "Optimized Layout"]
        violation_counts = [
            len(self.validate_layout(original_layout, constraints)),
            len(self.validate_layout(optimized_layout, constraints))
        ]
        
        room_width = original_layout['room']['width']
        room_height = original_layout['room']['height']
        
        # Color scheme
        colors = {
            'Bed': '#FF6B6B', 'Sofa': '#4ECDC4', 'Wardrobe': '#45B7D1',
            'Study Table': '#96CEB4', 'Study Chair': '#FFEAA7', 'Bedside Table': '#DDA0DD',
            'Room': '#F8F9FA'
        }
        
        for layout_idx, (layout, title) in enumerate(zip(layouts, titles)):
            # Layout view (top row)
            ax = [ax1, ax2][layout_idx]
            furniture = layout['furniture']
            violations = self.validate_layout(layout, constraints)
            
            # Draw room
            room_rect = Rectangle((0, 0), room_width, room_height,
                                linewidth=3, edgecolor='black', facecolor=colors['Room'])
            ax.add_patch(room_rect)
            
            # Draw furniture
            for item in furniture:
                color = colors.get(item['name'], '#CCCCCC')
                furniture_rect = Rectangle(
                    (item['x'], item['y']), item['width'], item['height'],
                    linewidth=2, edgecolor='black', facecolor=color, alpha=0.8
                )
                ax.add_patch(furniture_rect)
                
                # Label
                label_x = item['x'] + item['width'] / 2
                label_y = item['y'] + item['height'] / 2
                ax.text(label_x, label_y, item['name'], ha='center', va='center', 
                       fontsize=10, fontweight='bold')
            
            ax.set_xlim(-50, room_width + 50)
            ax.set_ylim(-50, room_height + 50)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{title}\nViolations: {violation_counts[layout_idx]}', 
                        fontsize=14, fontweight='bold')
            
            # Violations view (bottom row)
            ax_violations = [ax3, ax4][layout_idx]
            
            # Draw room
            room_rect = Rectangle((0, 0), room_width, room_height,
                                linewidth=3, edgecolor='black', facecolor=colors['Room'])
            ax_violations.add_patch(room_rect)
            
            # Draw furniture
            for item in furniture:
                color = colors.get(item['name'], '#CCCCCC')
                furniture_rect = Rectangle(
                    (item['x'], item['y']), item['width'], item['height'],
                    linewidth=2, edgecolor='black', facecolor=color, alpha=0.6
                )
                ax_violations.add_patch(furniture_rect)
            
            # Draw violation indicators
            min_clearance = constraints['room_constraints']['min_path_width']
            violation_count = 0
            
            for i, item1 in enumerate(furniture):
                for j, item2 in enumerate(furniture[i+1:], i+1):
                    distance = self._calculate_minimum_distance(item1, item2)
                    
                    if distance < min_clearance:
                        # Draw violation line
                        c1_x = item1['x'] + item1['width'] / 2
                        c1_y = item1['y'] + item1['height'] / 2
                        c2_x = item2['x'] + item2['width'] / 2
                        c2_y = item2['y'] + item2['height'] / 2
                        
                        ax_violations.plot([c1_x, c2_x], [c1_y, c2_y], 
                                         'r-', linewidth=3, alpha=0.8)
                        
                        # Distance label
                        mid_x = (c1_x + c2_x) / 2
                        mid_y = (c1_y + c2_y) / 2
                        ax_violations.text(mid_x, mid_y, f'{distance:.0f}cm', 
                                         bbox=dict(boxstyle="round,pad=0.2", 
                                                  facecolor="red", alpha=0.9),
                                         fontsize=9, ha='center', va='center', 
                                         color='white', fontweight='bold')
                        violation_count += 1
            
            ax_violations.set_xlim(-50, room_width + 50)
            ax_violations.set_ylim(-50, room_height + 50)
            ax_violations.set_aspect('equal')
            ax_violations.grid(True, alpha=0.3)
            ax_violations.set_title(f'{title} - Violation Analysis\nPath Violations: {violation_count}', 
                                  fontsize=12, fontweight='bold')
        
        # Add improvement summary
        improvement = violation_counts[0] - violation_counts[1]
        improvement_pct = (improvement / max(violation_counts[0], 1)) * 100
        
        summary_text = f"""OPTIMIZATION SUMMARY
        
Original violations: {violation_counts[0]}
Optimized violations: {violation_counts[1]}
Improvement: {improvement} violations resolved ({improvement_pct:.1f}%)

Status: {'✅ FULLY OPTIMIZED' if violation_counts[1] == 0 else f'🔄 PARTIALLY OPTIMIZED ({violation_counts[1]} remaining)'}"""
        
        fig.text(0.02, 0.02, summary_text, fontsize=12, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", 
                         facecolor="lightgreen" if violation_counts[1] == 0 else "lightyellow"))
        
        # Add legend
        legend_elements = [patches.Patch(facecolor=colors.get(name, '#CCCCCC'), 
                                       edgecolor='black', label=name) 
                          for name in colors.keys() if name != 'Room']
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, right=0.85)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison visualization saved to {output_path}")
    
    def save_layout(self, layout: Dict, output_path: str):
        """Save optimized layout to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(layout, f, indent=2)
        print(f"Optimized layout saved to {output_path}")
    
    def generate_optimization_report(self, original_layout: Dict, optimized_layout: Dict, constraints: Dict) -> Dict:
        """Generate comprehensive optimization report"""
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
                    'rotation_changed': orig_item.get('rotation', 0) != opt_item.get('rotation', 0),
                    'original_position': {'x': orig_item['x'], 'y': orig_item['y']},
                    'new_position': {'x': opt_item['x'], 'y': opt_item['y']}
                })
        
        report = {
            'optimization_summary': {
                'violations_before': len(original_violations),
                'violations_after': len(optimized_violations),
                'improvement': len(original_violations) - len(optimized_violations),
                'optimization_successful': len(optimized_violations) == 0,
                'improvement_percentage': round(((len(original_violations) - len(optimized_violations)) / max(len(original_violations), 1)) * 100, 2)
            },
            'original_violations': original_violations,
            'remaining_violations': optimized_violations,
            'furniture_movements': movements,
            'total_items_moved': len(movements),
            'recommendations': self._generate_recommendations(optimized_violations, constraints)
        }
        
        return report
    
    def _generate_recommendations(self, violations: List[str], constraints: Dict) -> List[str]:
        """Generate actionable recommendations based on remaining violations"""
        recommendations = []
        
        if any("path width" in v for v in violations):
            recommendations.append("Consider rearranging furniture to create wider pathways for wheelchair accessibility")
        
        if any("front clearance" in v for v in violations):
            recommendations.append("Ensure furniture with front access requirements has adequate space for approach")
        
        if any("Study Table" in v and "Study Chair" in v for v in violations):
            recommendations.append("Position study chair to allow easy access and exit from the desk area")
        
        if len(violations) > 0:
            recommendations.append("Consider removing or replacing some furniture items if space constraints persist")
            recommendations.append("Explore wall-mounted solutions for some items to free up floor space")
        
        return recommendations

# Enhanced main function
def main():
    # Initialize optimizer
    API_KEY = "AIzaSyBKXNL1iJw3vsIFo1crgGo25JbGfsaKzr8"  # Replace with your key
    optimizer = RoomLayoutOptimizer(API_KEY)
    
    # File paths
    constraints_path = r"constraints/enhanced_barrier_free_constraints.json"
    layout_path = r"Input-Layouts/room-layout-3.json"
    output_path = r"Outputs/FT/optimized_room_layout.json"
    report_path = r"Outputs/FT/optimization_report.json"
    original_viz_path = r"Outputs/Ft/original_layout_visualization.png"
    optimized_viz_path = r"Outputs/FT/optimized_layout_visualization.png"
    comparison_viz_path = r"Outputs/FT/layout_comparison.png"
    
    try:
        # Load data
        constraints = optimizer.load_constraints(constraints_path)
        original_layout = optimizer.load_layout(layout_path)
        
        print("="*60)
        print("BARRIER-FREE ROOM LAYOUT OPTIMIZER")
        print("="*60)
        print(f"Room dimensions: {original_layout['room']['width']}cm x {original_layout['room']['height']}cm")
        print(f"Furniture items: {len(original_layout['furniture'])}")
        
        # Visualize original layout
        print("\nGenerating original layout visualization...")
        optimizer.visualize_layout(original_layout, constraints, original_viz_path, "Original Layout")
        
        # Optimize layout
        print("\nStarting optimization process...")
        optimized_layout = optimizer.optimize_layout(original_layout, constraints, max_iterations=8)
        
        # Generate visualizations
        print("\nGenerating optimized layout visualization...")
        optimizer.visualize_layout(optimized_layout, constraints, optimized_viz_path, "Optimized Layout")
        
        # Create comparison visualization
        print("Creating comparison visualization...")
        optimizer.create_comparison_visualization(original_layout, optimized_layout, constraints, comparison_viz_path)
        
        # Generate comprehensive report
        print("Generating optimization report...")
        report = optimizer.generate_optimization_report(original_layout, optimized_layout, constraints)
        
        # Save results
        optimizer.save_layout(optimized_layout, output_path)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print detailed summary
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Violations before: {report['optimization_summary']['violations_before']}")
        print(f"Violations after: {report['optimization_summary']['violations_after']}")
        print(f"Improvement: {report['optimization_summary']['improvement']} violations resolved")
        print(f"Success rate: {report['optimization_summary']['improvement_percentage']}%")
        print(f"Items repositioned: {report['total_items_moved']}")
        print(f"Fully optimized: {'✅ YES' if report['optimization_summary']['optimization_successful'] else '❌ NO'}")
        
        if report['remaining_violations']:
            print(f"\nRemaining violations ({len(report['remaining_violations'])}):")
            for i, violation in enumerate(report['remaining_violations'][:5], 1):
                print(f"  {i}. {violation}")
            if len(report['remaining_violations']) > 5:
                print(f"  ... and {len(report['remaining_violations']) - 5} more")
        
        if report['recommendations']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print(f"\nFiles generated:")
        print(f"  📊 Original layout: {original_viz_path}")
        print(f"  📊 Optimized layout: {optimized_viz_path}")
        print(f"  📊 Comparison view: {comparison_viz_path}")
        print(f"  📄 Layout JSON: {output_path}")
        print(f"  📄 Full report: {report_path}")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please ensure all required files exist in the specified paths.")
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()