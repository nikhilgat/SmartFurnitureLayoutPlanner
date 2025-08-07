import numpy as np
import random
import json
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional

# Load constraints JSON
with open("barrierfreiRL/barrier_free_constraints.json") as f:
    constraints = json.load(f)

class EnhancedFurnitureEnv(gym.Env):
    def __init__(self, room_width=800, room_height=600, furniture_list=None, max_steps=1000, 
                 grid_size=10, use_discrete_actions=True):
        super().__init__()
        self.room_width = room_width
        self.room_height = room_height
        self.max_steps = max_steps
        self.current_step = 0
        self.grid_size = grid_size  # For discrete positioning
        self.use_discrete_actions = use_discrete_actions
        
        # Default furniture if none provided
        if furniture_list is None:
            self.furniture_templates = [
                {"name": "Table", "width": 120, "height": 70, "zHeight": 75},
                {"name": "Sofa", "width": 200, "height": 90, "zHeight": 85},
                {"name": "Chair", "width": 50, "height": 50, "zHeight": 90},
                {"name": "Bed", "width": 180, "height": 200, "zHeight": 55},
                {"name": "Wardrobe", "width": 150, "height": 60, "zHeight": 200}
            ]
        else:
            self.furniture_templates = furniture_list
            
        self.num_items = len(self.furniture_templates)
        self.furniture = self._initialize_furniture()
        
        # Door position
        self.door_position = {"x": 10, "y": room_height // 2, "width": 90, "height": 200}
        
        # Action space: [furniture_idx, x_pos, y_pos, rotation]
        if use_discrete_actions:
            # Discrete actions for more precise control
            x_positions = room_width // grid_size
            y_positions = room_height // grid_size
            rotations = len(constraints["rotation_angles"])
            
            self.action_space = spaces.MultiDiscrete([
                self.num_items,  # which furniture to move
                x_positions,     # x position (discretized)
                y_positions,     # y position (discretized)  
                rotations        # rotation options
            ])
        else:
            # Continuous actions (your current approach)
            self.action_space = spaces.Box(
                low=np.array([-1, -1, -1, -1]), 
                high=np.array([1, 1, 1, 1]), 
                dtype=np.float32
            )
        
        # Observation: normalized positions, sizes, rotations for all furniture
        obs_size = self.num_items * 6  # x, y, w, h, rotation, constraint_satisfaction
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
        
        # Track best layout found
        self.best_layout = None
        self.best_reward = float('-inf')
        self.layout_history = []
        
    def _initialize_furniture(self):
        """Initialize furniture with templates and random positions"""
        furniture = []
        for template in self.furniture_templates:
            furniture.append({
                "name": template["name"],
                "width": template["width"],
                "height": template["height"],
                "zHeight": template["zHeight"],
                "x": random.randint(50, self.room_width - template["width"] - 50),
                "y": random.randint(50, self.room_height - template["height"] - 50),
                "rotation": random.choice(constraints["rotation_angles"])
            })
        return furniture
    
    def _get_obs(self):
        """Get normalized observation including constraint satisfaction"""
        obs = []
        for f in self.furniture:
            # Normalized position and size
            obs.extend([
                f["x"] / self.room_width,
                f["y"] / self.room_height,
                f["width"] / self.room_width,
                f["height"] / self.room_height,
                f["rotation"] / 360.0,
                self._get_furniture_constraint_score(f)  # How well constraints are satisfied
            ])
        return np.array(obs, dtype=np.float32)
    
    def _get_furniture_constraint_score(self, furniture):
        """Calculate how well a single furniture item satisfies constraints (0-1)"""
        rules = constraints["furniture_specific_clearances"].get(furniture["name"], {})
        if not rules:
            return 1.0
            
        violations = 0
        total_checks = 0
        
        # Check clearances
        clearance_checks = [
            ("left_access", furniture["x"]),
            ("right_access", self.room_width - (furniture["x"] + furniture["width"])),
            ("front_clearance", self.room_height - (furniture["y"] + furniture["height"]))
        ]
        
        for rule_name, actual_clearance in clearance_checks:
            if rule_name in rules:
                total_checks += 1
                if actual_clearance < rules[rule_name]:
                    violations += 1
                    
        if "surround_clearance" in rules:
            clearance = rules["surround_clearance"]
            edges = [
                furniture["x"],
                self.room_width - (furniture["x"] + furniture["width"]),
                furniture["y"],
                self.room_height - (furniture["y"] + furniture["height"])
            ]
            total_checks += 4
            violations += sum(1 for edge in edges if edge < clearance)
        
        return 1.0 - (violations / max(total_checks, 1))
    
    def _calculate_enhanced_reward(self):
        """Enhanced reward function for optimal placement"""
        base_reward = 1000.0  # Start with high base reward
        
        # 1. Constraint satisfaction (most important)
        constraint_score = 0.0
        for furniture in self.furniture:
            constraint_score += self._get_furniture_constraint_score(furniture)
        constraint_reward = (constraint_score / len(self.furniture)) * 500
        
        # 2. Collision penalty (critical)
        collision_penalty = 0.0
        for i in range(len(self.furniture)):
            for j in range(i + 1, len(self.furniture)):
                if self._check_collision(self.furniture[i], self.furniture[j]):
                    collision_penalty += 200  # Heavy penalty
        
        # 3. Accessibility requirements
        accessibility_reward = 0.0
        if self._check_doorway_clear():
            accessibility_reward += 100
        if self._check_turning_area_clear():
            accessibility_reward += 100
        
        # 4. Space efficiency bonus
        space_efficiency = self._calculate_space_efficiency()
        efficiency_reward = space_efficiency * 100
        
        # 5. Aesthetic arrangement bonus (furniture spacing)
        aesthetic_bonus = self._calculate_aesthetic_score() * 50
        
        total_reward = (base_reward + constraint_reward + accessibility_reward + 
                       efficiency_reward + aesthetic_bonus - collision_penalty)
        
        return max(-500, min(2000, total_reward))  # Clamp reward
    
    def _calculate_space_efficiency(self):
        """Calculate how efficiently space is used"""
        total_furniture_area = sum(f["width"] * f["height"] for f in self.furniture)
        room_area = self.room_width * self.room_height
        return min(total_furniture_area / room_area, 0.6)  # Cap at 60% for reasonable layouts
    
    def _calculate_aesthetic_score(self):
        """Simple aesthetic scoring based on furniture spacing"""
        if len(self.furniture) < 2:
            return 1.0
            
        distances = []
        for i in range(len(self.furniture)):
            for j in range(i + 1, len(self.furniture)):
                f1, f2 = self.furniture[i], self.furniture[j]
                center1 = (f1["x"] + f1["width"]/2, f1["y"] + f1["height"]/2)
                center2 = (f2["x"] + f2["width"]/2, f2["y"] + f2["height"]/2)
                dist = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                distances.append(dist)
        
        # Prefer moderate distances (not too cramped, not too spread out)
        avg_distance = np.mean(distances)
        ideal_distance = 150  # Adjust based on room size
        return 1.0 - abs(avg_distance - ideal_distance) / ideal_distance
    
    def _check_collision(self, item1, item2):
        """Check if two furniture items collide"""
        return not (item1["x"] + item1["width"] <= item2["x"] or
                   item2["x"] + item2["width"] <= item1["x"] or
                   item1["y"] + item1["height"] <= item2["y"] or
                   item2["y"] + item2["height"] <= item1["y"])
    
    def _check_doorway_clear(self):
        """Check if doorway is clear"""
        for f in self.furniture:
            if self._check_collision(f, self.door_position):
                return False
        return True
    
    def _check_turning_area_clear(self):
        """Check if wheelchair turning area is clear"""
        cx, cy = self.room_width // 2, self.room_height // 2
        radius = constraints["clearance_requirements"]["turning_area_wheelchair"][0] // 2
        
        for f in self.furniture:
            # Check if furniture overlaps with turning circle
            fx_min, fx_max = f["x"], f["x"] + f["width"]
            fy_min, fy_max = f["y"], f["y"] + f["height"]
            
            # Simple circle-rectangle intersection check
            closest_x = max(fx_min, min(cx, fx_max))
            closest_y = max(fy_min, min(cy, fy_max))
            
            distance = ((closest_x - cx)**2 + (closest_y - cy)**2)**0.5
            if distance < radius:
                return False
        return True
    
    def _is_perfect_layout(self):
        """Check if current layout satisfies all constraints perfectly"""
        # No collisions
        for i in range(len(self.furniture)):
            for j in range(i + 1, len(self.furniture)):
                if self._check_collision(self.furniture[i], self.furniture[j]):
                    return False
        
        # All furniture constraint scores are 1.0
        for furniture in self.furniture:
            if self._get_furniture_constraint_score(furniture) < 0.99:
                return False
        
        # Accessibility requirements met
        return self._check_doorway_clear() and self._check_turning_area_clear()
    
    def step(self, action):
        self.current_step += 1
        
        if self.use_discrete_actions:
            furniture_idx = action[0]
            x_grid = action[1]
            y_grid = action[2]
            rot_idx = action[3]
            
            # Convert grid positions to actual coordinates
            new_x = x_grid * self.grid_size
            new_y = y_grid * self.grid_size
            new_rotation = constraints["rotation_angles"][rot_idx]
            
        else:
            # Your original continuous action processing
            furniture_idx = int(np.clip((action[0] + 1) / 2 * (self.num_items - 1), 0, self.num_items - 1))
            dx = int(action[1] * 20)  # Increased movement range
            dy = int(action[2] * 20)
            rot_idx = int(np.clip((action[3] + 1) / 2 * (len(constraints["rotation_angles"]) - 1), 
                                 0, len(constraints["rotation_angles"]) - 1))
            
            furniture = self.furniture[furniture_idx]
            new_x = np.clip(furniture["x"] + dx, 0, self.room_width - furniture["width"])
            new_y = np.clip(furniture["y"] + dy, 0, self.room_height - furniture["height"])
            new_rotation = constraints["rotation_angles"][rot_idx]
        
        # Apply action
        furniture = self.furniture[furniture_idx]
        furniture["x"] = int(np.clip(new_x, 0, self.room_width - furniture["width"]))
        furniture["y"] = int(np.clip(new_y, 0, self.room_height - furniture["height"]))
        furniture["rotation"] = new_rotation
        
        # Calculate reward
        reward = self._calculate_enhanced_reward()
        
        # Track best layout
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_layout = self.get_layout_json()
        
        # Check if perfect layout achieved
        perfect = self._is_perfect_layout()
        if perfect:
            reward += 1000  # Bonus for perfect layout
        
        # Episode termination
        done = self.current_step >= self.max_steps or perfect
        
        info = {
            "reward": reward,
            "perfect_layout": perfect,
            "best_reward": self.best_reward,
            "constraint_satisfaction": np.mean([self._get_furniture_constraint_score(f) for f in self.furniture])
        }
        
        return self._get_obs(), reward, done, False, info
    
    def get_layout_json(self):
        """Get current layout in your desired JSON format"""
        return {
            "room": {
                "width": self.room_width,
                "height": self.room_height
            },
            "furniture": [
                {
                    "name": f["name"],
                    "x": f["x"],
                    "y": f["y"],
                    "width": f["width"],
                    "height": f["height"],
                    "zHeight": str(f["zHeight"]),
                    "rotation": f["rotation"]
                }
                for f in self.furniture
            ],
            "openings": []  # Add door/window info if needed
        }
    
    def get_best_layout(self):
        """Get the best layout found during training"""
        return self.best_layout if self.best_layout else self.get_layout_json()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.furniture = self._initialize_furniture()
        self.current_step = 0
        return self._get_obs(), {}
    
    def render(self, mode='human', save_path=None):
        """Enhanced rendering with constraint visualization"""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, self.room_width)
        ax.set_ylim(0, self.room_height)
        ax.set_aspect('equal')
        ax.set_title(f"Step: {self.current_step}, Reward: {self._calculate_enhanced_reward():.1f}")
        
        # Room boundary
        room_rect = patches.Rectangle((0, 0), self.room_width, self.room_height,
                                    linewidth=3, edgecolor='black', facecolor='none')
        ax.add_patch(room_rect)
        
        # Turning area
        turn_radius = constraints["clearance_requirements"]["turning_area_wheelchair"][0] / 2
        circle = patches.Circle((self.room_width / 2, self.room_height / 2), turn_radius,
                              edgecolor='green', facecolor='lightgreen', alpha=0.3, 
                              linestyle='--', linewidth=2, label='Turning Area')
        ax.add_patch(circle)
        
        # Door
        door = self.door_position
        door_rect = patches.Rectangle((door["x"], door["y"]), door["width"], door["height"],
                                    edgecolor='brown', facecolor='burlywood', label='Door')
        ax.add_patch(door_rect)
        
        # Furniture with constraint satisfaction coloring
        for f in self.furniture:
            constraint_score = self._get_furniture_constraint_score(f)
            
            # Color based on constraint satisfaction (red=bad, green=good)
            if constraint_score >= 0.9:
                color = 'lightgreen'
                edge_color = 'green'
            elif constraint_score >= 0.7:
                color = 'yellow' 
                edge_color = 'orange'
            else:
                color = 'lightcoral'
                edge_color = 'red'
            
            rect = patches.Rectangle((f["x"], f["y"]), f["width"], f["height"],
                                   angle=f["rotation"], edgecolor=edge_color, 
                                   facecolor=color, alpha=0.7, linewidth=2)
            ax.add_patch(rect)
            
            # Label
            ax.text(f["x"] + f["width"] / 2, f["y"] + f["height"] / 2, 
                   f"{f['name']}\n{constraint_score:.2f}", 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()