import json
import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FurnitureItem:
    """Represents a piece of furniture with accessibility constraints"""
    name: str
    width: float
    height: float
    z_height: float
    x: float = 0.0
    y: float = 0.0
    rotation: int = 0  # 0, 90, 180, 270
    is_fixed: bool = False
    accessibility_type: str = "standard"  # "wheelchair", "mobility_aid", "standard"

@dataclass
class AccessibilityConstraints:
    """Holds all accessibility requirements"""
    general_clearance: float = 90
    wheelchair_turning_radius: float = 150
    path_width: float = 120
    furniture_clearances: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.furniture_clearances is None:
            self.furniture_clearances = {}

class BarrierFreeEnvironment(gym.Env):
    """
    Gymnasium environment for barrier-free furniture arrangement
    
    Observation Space: Flattened furniture positions and rotations
    Action Space: Select furniture, move direction, rotation
    """
    
    def __init__(self, 
                 room_layout: Dict,
                 constraints: Dict,
                 accessibility_mode: str = "wheelchair"):
        super().__init__()
        
        self.room_width = room_layout["room"]["width"]
        self.room_height = room_layout["room"]["height"]
        self.accessibility_mode = accessibility_mode
        
        # Load constraints
        self.constraints = self._load_constraints(constraints)
        
        # Initialize furniture
        self.furniture_items = self._create_furniture_items(room_layout["furniture"])
        self.openings = room_layout.get("openings", [])
        
        # RL Environment setup
        self.max_steps = 500
        self.current_step = 0
        
        # Action space: [furniture_id, action_type, direction/rotation]
        self.action_space = spaces.MultiDiscrete([
            len(self.furniture_items),  # which furniture
            2,                          # move or rotate
            4                           # direction/rotation
        ])
        
        # Observation space: [x, y, rotation] for each furniture + room info
        obs_size = len(self.furniture_items) * 3 + 4
        self.observation_space = spaces.Box(
            low=0, high=max(self.room_width, self.room_height, 360, self.max_steps),
            shape=(obs_size,), dtype=np.float32
        )
        
        # Tracking
        self.best_score = float('-inf')
        self.episode_rewards = []
        self.violation_history = []
        
        # Movement step size
        self.move_step = 10
        
        logger.info(f"Environment initialized: {self.room_width}x{self.room_height}")
        logger.info(f"Furniture count: {len(self.furniture_items)}")
        logger.info(f"Accessibility mode: {accessibility_mode}")
    
    def _load_constraints(self, constraints_dict: Dict) -> AccessibilityConstraints:
        """Load accessibility constraints from dictionary"""
        clearances = constraints_dict.get("clearance_requirements", {})
        furniture_clearances = constraints_dict.get("furniture_specific_clearances", {})
        
        return AccessibilityConstraints(
            general_clearance=clearances.get("general_clearance", 90),
            wheelchair_turning_radius=clearances.get("turning_area_wheelchair", [150, 150])[0],
            path_width=clearances.get("path_width", 120),
            furniture_clearances=furniture_clearances
        )
    
    def _create_furniture_items(self, furniture_list: List[Dict]) -> List[FurnitureItem]:
        """Create furniture objects from room layout"""
        items = []
        for i, furniture in enumerate(furniture_list):
            z_height = furniture.get("zHeight", furniture.get("z_height", 75))
            if isinstance(z_height, str):
                z_height = float(z_height)
            
            item = FurnitureItem(
                name=furniture["name"],
                width=furniture["width"],
                height=furniture["height"],
                z_height=z_height,
                x=furniture.get("x", 50),
                y=furniture.get("y", 50),
                rotation=furniture.get("rotation", 0)
            )
            items.append(item)
        
        return items
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        for item in self.furniture_items:
            if not item.is_fixed:
                item.x += random.uniform(-20, 20)
                item.y += random.uniform(-20, 20)
                item.x = np.clip(item.x, 0, self.room_width - item.width)
                item.y = np.clip(item.y, 0, self.room_height - item.height)
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment state"""
        obs = []
        
        for item in self.furniture_items:
            obs.extend([
                item.x / self.room_width,
                item.y / self.room_height,
                item.rotation / 360.0
            ])
        
        obs.extend([
            self.room_width / 1000.0,
            self.room_height / 1000.0,
            self.current_step / self.max_steps,
            len(self.furniture_items) / 10.0
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Execute one step in the environment"""
        furniture_idx, action_type, direction = action
        furniture_idx = int(furniture_idx)
        action_type = int(action_type)
        direction = int(direction)
        
        if self.furniture_items[furniture_idx].is_fixed:
            reward = -2 # was -1 
        else:
            reward = self._execute_action(furniture_idx, action_type, direction)
        
        self.current_step += 1
        
        done = self.current_step >= self.max_steps
        
        total_reward = reward + self._calculate_layout_quality()
        total_reward = np.clip(total_reward, -10, 10)
        
        info = {
            'violations': self._count_violations(),
            'accessibility_score': self._calculate_accessibility_score(),
            'layout_efficiency': self._calculate_layout_efficiency(),
            'step': self.current_step
        }
        
        return self._get_observation(), total_reward, done, False, info
    
    def _execute_action(self, furniture_idx: int, action_type: int, direction: int) -> float:
        """Execute the action and return immediate reward"""
        item = self.furniture_items[furniture_idx]
        old_x, old_y, old_rot = item.x, item.y, item.rotation
        
        if action_type == 0:  # Move
            if direction == 0:    # Up
                item.y = max(0, item.y - self.move_step)
            elif direction == 1:  # Down
                item.y = min(self.room_height - item.height, item.y + self.move_step)
            elif direction == 2:  # Left
                item.x = max(0, item.x - self.move_step)
            elif direction == 3:  # Right
                item.x = min(self.room_width - item.width, item.x + self.move_step)
        
        elif action_type == 1:  # Rotate
            item.rotation = (direction * 90) % 360
            if direction in [1, 3]:
                item.width, item.height = item.height, item.width
        
        old_violations = self._count_violations_for_positions([(old_x, old_y, old_rot, furniture_idx)])
        new_violations = self._count_violations()
        
        if new_violations < old_violations:
            return 2.0
        elif new_violations == old_violations:
            return 0.0
        else:
            return -1.0
    
    def _count_violations(self) -> int:
        """Count accessibility violations in current layout"""
        violations = 0
        
        for i, item1 in enumerate(self.furniture_items):
            for j, item2 in enumerate(self.furniture_items[i+1:], i+1):
                if self._items_overlap(item1, item2):
                    violations += 1
        
        for item in self.furniture_items:
            violations += self._check_clearance_violations(item)
        
        if not self._check_path_connectivity():
            violations += 5
        
        return violations
    
    def _count_violations_for_positions(self, position_list: List[Tuple]) -> int:
        """Count violations for specific positions (for reward calculation)"""
        original_positions = [(item.x, item.y, item.rotation) for item in self.furniture_items]
        
        for x, y, rot, idx in position_list:
            self.furniture_items[idx].x = x
            self.furniture_items[idx].y = y
            self.furniture_items[idx].rotation = rot
        
        violations = self._count_violations()
        
        for i, (x, y, rot) in enumerate(original_positions):
            self.furniture_items[i].x = x
            self.furniture_items[i].y = y
            self.furniture_items[i].rotation = rot
        
        return violations
    
    def _items_overlap(self, item1: FurnitureItem, item2: FurnitureItem) -> bool:
        """Check if two furniture items overlap"""
        return not (item1.x + item1.width <= item2.x or
                   item2.x + item2.width <= item1.x or
                   item1.y + item1.height <= item2.y or
                   item2.y + item2.height <= item1.y)
    
    def _check_clearance_violations(self, item: FurnitureItem) -> int:
        """Check clearance violations for a furniture item"""
        violations = 0
        
        clearances = self.constraints.furniture_clearances.get(item.name, {})
        
        for other_item in self.furniture_items:
            if other_item == item:
                continue
            
            distance = self._calculate_distance(item, other_item)
            required_clearance = clearances.get("front_clearance", 
                                               self.constraints.general_clearance)
            
            if distance < required_clearance:
                violations += 1
        
        return violations
    
    def _calculate_distance(self, item1: FurnitureItem, item2: FurnitureItem) -> float:
        """Calculate minimum distance between two furniture items"""
        center1_x = item1.x + item1.width / 2
        center1_y = item1.y + item1.height / 2
        center2_x = item2.x + item2.width / 2
        center2_y = item2.y + item2.height / 2
        
        dx = abs(center1_x - center2_x) - (item1.width + item2.width) / 2
        dy = abs(center1_y - center2_y) - (item1.height + item2.height) / 2
        
        return max(0, max(dx, dy))
    
    def _check_path_connectivity(self) -> bool:
        """Check if all areas are connected by wheelchair-accessible paths"""
        for opening in self.openings:
            wall = opening.get("wall")
            position = opening.get("position", self.room_width // 2)
            if wall == "bottom":
                entrance_x, entrance_y = position, 0
            elif wall == "top":
                entrance_x, entrance_y = position, self.room_height
            elif wall == "left":
                entrance_x, entrance_y = 0, position
            elif wall == "right":
                entrance_x, entrance_y = self.room_width, position
            else:
                entrance_x, entrance_y = self.room_width // 2, 0
            
            for item in self.furniture_items:
                if item.name in ["Bed", "Sofa", "Table"]:
                    if not self._has_clear_path(entrance_x, entrance_y, 
                                              item.x + item.width/2, 
                                              item.y + item.height/2):
                        return False
        return True
    
    def _has_clear_path(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if there's a clear path between two points"""
        path_width = self.constraints.path_width
        
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            
            for item in self.furniture_items:
                if (px >= item.x - path_width/2 and px <= item.x + item.width + path_width/2 and
                    py >= item.y - path_width/2 and py <= item.y + item.height + path_width/2):
                    return False
        
        return True
    
    def _calculate_layout_quality(self) -> float:
        """Calculate overall layout quality score"""
        score = 0.0
        
        accessibility = self._calculate_accessibility_score()
        score += accessibility * 3
        
        efficiency = self._calculate_layout_efficiency()
        score += efficiency
        
        relationships = self._calculate_relationship_score()
        score += relationships
        
        return score
    
    def _calculate_accessibility_score(self) -> float:
        """Calculate accessibility compliance score (0-1)"""
        total_checks = 0
        passed_checks = 0
        
        for item in self.furniture_items:
            clearances = self.constraints.furniture_clearances.get(item.name, {})
            for clearance_type, required_distance in clearances.items():
                total_checks += 1
                if self._check_specific_clearance(item, clearance_type, required_distance):
                    passed_checks += 1
        
        turning_areas = self._find_turning_areas()
        for area in turning_areas:
            total_checks += 1
            if area["width"] >= self.constraints.wheelchair_turning_radius and \
               area["height"] >= self.constraints.wheelchair_turning_radius:
                passed_checks += 1
        
        return passed_checks / max(total_checks, 1)
    
    def _check_specific_clearance(self, item: FurnitureItem, 
                                clearance_type: str, required_distance: float) -> bool:
        """Check specific clearance requirement"""
        if clearance_type == "front_clearance":
            front_x = item.x
            front_y = item.y - required_distance
            front_width = item.width
            front_height = required_distance
            
            for other_item in self.furniture_items:
                if other_item == item:
                    continue
                if self._rectangles_overlap(front_x, front_y, front_width, front_height,
                                          other_item.x, other_item.y, 
                                          other_item.width, other_item.height):
                    return False
            return True
        
        return True
    
    def _rectangles_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2) -> bool:
        """Check if two rectangles overlap"""
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)
    
    def _find_turning_areas(self) -> List[Dict]:
        """Find available turning areas in the room"""
        areas = []
        grid_size = 30
        for x in range(0, int(self.room_width - 150), grid_size):
            for y in range(0, int(self.room_height - 150), grid_size):
                if self._is_area_clear(x, y, 150, 150):
                    areas.append({"x": x, "y": y, "width": 150, "height": 150})
        
        return areas
    
    def _is_area_clear(self, x: float, y: float, width: float, height: float) -> bool:
        """Check if an area is clear of furniture"""
        for item in self.furniture_items:
            if self._rectangles_overlap(x, y, width, height,
                                      item.x, item.y, item.width, item.height):
                return False
        return True
    
    def _calculate_layout_efficiency(self) -> float:
        """Calculate how efficiently space is used"""
        total_furniture_area = sum(item.width * item.height for item in self.furniture_items)
        room_area = self.room_width * self.room_height
        coverage_ratio = total_furniture_area / room_area
        
        optimal_coverage = 0.35
        efficiency = 1.0 - abs(coverage_ratio - optimal_coverage) / optimal_coverage
        
        return max(0, efficiency)
    
    def _calculate_relationship_score(self) -> float:
        """Calculate functional relationship score"""
        score = 0.0
        relationships_checked = 0
        
        bed = next((item for item in self.furniture_items if "Bed" in item.name), None)
        bedside = next((item for item in self.furniture_items if "Bedside" in item.name), None)
        
        if bed and bedside:
            distance = self._calculate_distance(bed, bedside)
            if 30 <= distance <= 80:
                score += 1.0
            relationships_checked += 1
        
        sofa = next((item for item in self.furniture_items if "Sofa" in item.name), None)
        coffee_table = next((item for item in self.furniture_items 
                           if "Coffee" in item.name or "Table" in item.name), None)
        
        if sofa and coffee_table:
            distance = self._calculate_distance(sofa, coffee_table)
            if 40 <= distance <= 100:
                score += 1.0
            relationships_checked += 1
        
        return score / max(relationships_checked, 1)
    
    def get_layout_json(self) -> Dict:
        """Export current layout as JSON"""
        furniture_list = []
        for item in self.furniture_items:
            furniture_list.append({
                "name": item.name,
                "x": float(item.x),
                "y": float(item.y),
                "width": float(item.width),
                "height": float(item.height),
                "zHeight": float(item.z_height),
                "rotation": int(item.rotation)
            })
        
        return {
            "room": {
                "width": self.room_width,
                "height": self.room_height
            },
            "furniture": furniture_list,
            "openings": self.openings,
            "accessibility_score": self._calculate_accessibility_score(),
            "layout_efficiency": self._calculate_layout_efficiency(),
            "total_violations": self._count_violations()
        }
    
    def visualize_layout(self, save_path: Optional[str] = None):
        """Visualize the current layout"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        
        # Draw room
        room_rect = Rectangle((0, 0), self.room_width, self.room_height, 
                            linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(room_rect)
        
        # Draw furniture
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.furniture_items)))
        for i, item in enumerate(self.furniture_items):
            rect = Rectangle((item.x, item.y), item.width, item.height,
                           linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.7)
            ax.add_patch(rect)
            
            # Add label
            ax.text(item.x + item.width/2, item.y + item.height/2, item.name,
                   ha='center', va='center', fontsize=8, weight='bold')
        
        # Draw openings
        for opening in self.openings:
            wall = opening.get("wall")
            position = opening.get("position", 0)
            size = opening.get("size", 90)
            
            thickness = 10
            
            if wall == "bottom":
                x, y = position, 0
                width, height = size, thickness
            elif wall == "top":
                x, y = position, self.room_height - thickness
                width, height = size, thickness
            elif wall == "left":
                x, y = 0, position
                width, height = thickness, size
            elif wall == "right":
                x, y = self.room_width - thickness, position
                width, height = thickness, size
            else:
                logger.warning(f"Unrecognized wall type: {wall}")
                continue
            
            opening_rect = Rectangle((x, y), width, height,
                                   linewidth=2, edgecolor='red', facecolor='red', alpha=0.5)
            ax.add_patch(opening_rect)
        
        ax.set_xlim(0, self.room_width)
        ax.set_ylim(0, self.room_height)
        ax.set_aspect('equal')
        ax.set_title(f'Barrier-Free Layout (Accessibility Score: {self._calculate_accessibility_score():.2f})')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

def test_trained_model(model_path: str, 
                      room_layout_path: str, 
                      constraints_path: str,
                      num_episodes: int = 5):
    """Test the trained model and generate optimized layouts"""
    with open(room_layout_path, 'r') as f:
        room_layout = json.load(f)
    
    with open(constraints_path, 'r') as f:
        constraints = json.load(f)
    
    env = BarrierFreeEnvironment(room_layout, constraints)
    
    model = PPO.load(model_path, env=env, weights_only=True)
    
    best_layout = None
    best_score = float('-inf')
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        layout = env.get_layout_json()
        score = layout["accessibility_score"]
        
        logger.info(f"Episode {episode + 1}: Score = {score:.3f}, "
                   f"Violations = {layout['total_violations']}, "
                   f"Efficiency = {layout['layout_efficiency']:.3f}")
        
        if score > best_score:
            best_score = score
            best_layout = layout
    
    if best_layout:
        output_path = "optimized_barrier_free_layout.json"
        with open(output_path, 'w') as f:
            json.dump(best_layout, f, indent=2)
        logger.info(f"Best layout saved to {output_path}")
        
        env.furniture_items = env._create_furniture_items(best_layout["furniture"])
        env.visualize_layout("best_barrier_free_layout.png")
    
    return best_layout

if __name__ == "__main__":
    logger.info("Starting inference with trained model...")
    
    best_layout = test_trained_model(
        model_path="barrier_free_furniture_model.zip",
        room_layout_path="room-layout-new.json", 
        constraints_path="merged_barrier_free_constraints.json",
        num_episodes=10
    )
    
    if best_layout:
        logger.info("=== BARRIER-FREE LAYOUT OPTIMIZATION COMPLETE ===")
        logger.info(f"Final accessibility score: {best_layout['accessibility_score']:.3f}")
        logger.info(f"Layout efficiency: {best_layout['layout_efficiency']:.3f}")
        logger.info(f"Total violations: {best_layout['total_violations']}")
        logger.info("Optimized layout saved to: optimized_barrier_free_layout.json")
    else:
        logger.warning("No valid layout generated")