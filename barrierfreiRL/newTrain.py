# improved_train.py
# Enhanced version with better exploration, curriculum learning, and reward shaping

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque
import time
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# =========================
# IMPROVED CONFIG
# =========================
CONFIG = {
    "CONSTRAINTS_FILE": "layout_app_constraints.json",
    "PLANNER_LAYOUT_FILE": "room-layout-new.json",
    "UNITS_SCALE": 1.0,
    "SEED": 42,
    "TOTAL_TIMESTEPS": 100_000,  # Increased training time
    "MAX_STEPS_PER_EPISODE": 300,  # Longer episodes
    "SAVE_PATH": "ppo_barrier_free_improved",
    "RUN_EVAL": True,
    "EVAL_STEPS": 500,
    "OPTIMIZED_LAYOUT_OUT": "optimized_layout_improved.json",
    "CURRICULUM_STAGES": 3,  # Curriculum learning stages
} 


# =========================
# Data Classes (same as before)
# =========================
@dataclass
class FurnitureItem:
    name: str
    width: float
    depth: float
    height: float
    x: float
    y: float
    rotation: int
    movable: bool = True

    def get_rect(self) -> Tuple[float, float, float, float]:
        """Get axis-aligned bounding box."""
        if self.rotation % 180 == 0:
            return (self.x, self.y, self.width, self.depth)
        else:
            return (self.x, self.y, self.depth, self.width)

    def get_center(self) -> Tuple[float, float]:
        x, y, w, d = self.get_rect()
        return (x + w / 2.0, y + d / 2.0)

    def rotate_90(self, allowed_angles: List[int]):
        if not allowed_angles:
            return
        try:
            current_idx = allowed_angles.index(self.rotation)
            next_idx = (current_idx + 1) % len(allowed_angles)
            self.rotation = allowed_angles[next_idx]
        except ValueError:
            self.rotation = allowed_angles[0] if allowed_angles else 0


# =========================
# Utility Functions
# =========================
def rects_overlap(rect1: Tuple[float, float, float, float], 
                 rect2: Tuple[float, float, float, float]) -> bool:
    """Check if two rectangles overlap."""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def clamp_value(x: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, x))


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# =========================
# Curriculum Learning Callback
# =========================
class CurriculumCallback(BaseCallback):
    """Callback to implement curriculum learning."""
    
    def __init__(self, env_wrapper, total_timesteps, stages=3, verbose=0):
        super().__init__(verbose)
        self.env_wrapper = env_wrapper
        self.total_timesteps = total_timesteps
        self.stages = stages
        self.stage_duration = total_timesteps // stages
        self.current_stage = 0
        
    def _on_step(self) -> bool:
        # Update curriculum stage based on progress
        new_stage = min(self.stages - 1, self.num_timesteps // self.stage_duration)
        if new_stage != self.current_stage:
            self.current_stage = new_stage
            # Update environment difficulty
            if hasattr(self.env_wrapper, 'set_difficulty'):
                self.env_wrapper.set_difficulty(self.current_stage)
            if self.verbose > 0:
                print(f"Curriculum stage updated to {self.current_stage}")
        return True


# =========================
# Enhanced Environment
# =========================
class EnhancedBarrierFreeEnv(gym.Env):
    """
    Enhanced environment with better exploration, curriculum learning, and improved reward shaping.
    """
    
    metadata = {"render_modes": []}

    def __init__(self, 
                 constraints: Dict,
                 room_layout: Optional[Dict] = None,
                 seed: int = 0,
                 max_steps: int = 300,
                 initial_items: Optional[List[Dict]] = None,
                 difficulty_stage: int = 0):
        
        super().__init__()
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        self.rng = random.Random(seed)

        # Parse room dimensions
        if room_layout:
            self.room_width = float(room_layout["room"]["width"])
            self.room_height = float(room_layout["room"]["height"])
            door_data = room_layout["door_rect"]
            self.door_rect = tuple(float(x) for x in door_data)
        else:
            self.room_width = 800.0
            self.room_height = 600.0
            self.door_rect = (355.0, 0.0, 90.0, 5.0)

        # Parse constraints
        clear_reqs = constraints.get("clearance_requirements", {})
        furn_clear = constraints.get("furniture_specific_clearances", {})
        
        self.path_width = clear_reqs.get("path_width", 120.0)
        self.general_clearance = clear_reqs.get("general_clearance", 90.0)
        self.turning_diameter = 150.0
        self.step_size = 15.0  # Larger step size for faster movement
        self.allowed_rotations = [0, 90, 180, 270]

        # Door keep-clear area
        dx, dy, dw, dh = self.door_rect
        self.door_clear_rect = (dx, dy + dh, dw, 122.0)

        # Furniture catalog
        self.furniture_catalog = {
            "Bed": {"width": 180, "depth": 200, "height": 55},
            "Sofa": {"width": 200, "depth": 90, "height": 85},
            "Study Table": {"width": 120, "depth": 70, "height": 75},
            "Study Chair": {"width": 50, "depth": 50, "height": 90},
            "Wardrobe": {"width": 150, "depth": 60, "height": 200},
            "Bedside Table": {"width": 45, "depth": 45, "height": 60},
        }

        # Clearance requirements
        self.clearance_specs = furn_clear
        
        # Relationship requirements
        relationships = constraints.get("furniture_relationships", {})
        self.functional_pairs = relationships.get("functional_pairs", {})

        # Curriculum learning
        self.difficulty_stage = difficulty_stage
        self.max_difficulty = 2

        # Initialize furniture
        self.items: List[FurnitureItem] = []
        if initial_items:
            self._create_items_from_data(initial_items)
        else:
            self._create_curriculum_layout()

        # Fixed observation space for all possible items (max 6 items)
        max_items = 6
        # Enhanced action space: directional moves + rotation + smart actions
        num_items = len(self.items)
        # 4 moves + 1 rotate + 2 smart actions (move to center, move to wall) per item
        self.action_space = spaces.Discrete(max_items * 7)  # Use max_items for consistent action space
        
        # Fixed observation space for maximum possible items
        obs_size = max_items * 6 + 8  # +6 per item, +8 global features
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Episode tracking with enhanced metrics
        self.max_steps = max_steps
        self.current_step = 0
        self.best_reward_so_far = float('-inf')
        self.episode_violations = []
        self.exploration_bonus_decay = 0.995
        self.exploration_bonus = 1.0
        
        # Track visited states for exploration bonus
        self.state_visits = {}

    def set_difficulty(self, stage: int):
        """Set curriculum difficulty stage."""
        self.difficulty_stage = min(stage, self.max_difficulty)

    def _create_items_from_data(self, items_data: List[Dict]):
        """Create furniture items from provided data with some randomization."""
        name_map = {"Table": "Study Table", "Chair": "Study Chair"}
        
        for item_data in items_data:
            name = item_data["name"]
            name = name_map.get(name, name)
            
            if name not in self.furniture_catalog:
                continue
                
            catalog_item = self.furniture_catalog[name]
            
            # Add some randomization to initial positions based on difficulty
            randomization_factor = 0.1 if self.difficulty_stage == 0 else 0.3
            
            base_x = float(item_data["x"])
            base_y = float(item_data["y"])
            
            # Add random offset
            max_offset = min(50.0, self.room_width * randomization_factor)
            offset_x = self.rng.uniform(-max_offset, max_offset)
            offset_y = self.rng.uniform(-max_offset, max_offset)
            
            item = FurnitureItem(
                name=name,
                width=float(item_data["width"]),
                depth=float(item_data["depth"]),
                height=catalog_item["height"],
                x=base_x + offset_x,
                y=base_y + offset_y,
                rotation=int(item_data.get("rotation", 0)),
                movable=True
            )
            
            # Ensure item fits in room
            rect = item.get_rect()
            item.x = clamp_value(item.x, 0, self.room_width - rect[2])
            item.y = clamp_value(item.y, 0, self.room_height - rect[3])
            
            self.items.append(item)

    def _create_curriculum_layout(self):
        """Create initial layout based on curriculum stage."""
        items_to_place = ["Bed", "Sofa", "Study Table", "Study Chair", "Wardrobe", "Bedside Table"]
        
        # Reduce number of items for easier stages
        if self.difficulty_stage == 0:
            items_to_place = items_to_place[:3]  # Only 3 items
        elif self.difficulty_stage == 1:
            items_to_place = items_to_place[:5]  # 5 items
        
        for name in items_to_place:
            catalog_item = self.furniture_catalog[name]
            
            # Better initial placement strategy
            placed = False
            for attempt in range(200):  # More attempts
                rotation = self.rng.choice(self.allowed_rotations)
                width = catalog_item["width"]
                depth = catalog_item["depth"]
                
                if rotation % 180 == 90:
                    width, depth = depth, width
                
                # Smart placement: prefer areas away from door and other furniture
                if attempt < 100:
                    # First half: try strategic placement
                    if name == "Bed":
                        # Beds prefer corners
                        x = self.rng.choice([0, self.room_width - width])
                        y = self.rng.uniform(100, self.room_height - depth)
                    elif name == "Wardrobe":
                        # Wardrobes prefer walls
                        if self.rng.random() < 0.5:
                            x = self.rng.choice([0, self.room_width - width])
                            y = self.rng.uniform(0, self.room_height - depth)
                        else:
                            x = self.rng.uniform(0, self.room_width - width)
                            y = self.rng.choice([0, self.room_height - depth])
                    else:
                        # Other items prefer center areas
                        x = self.rng.uniform(width, self.room_width - width * 2)
                        y = self.rng.uniform(depth, self.room_height - depth * 2)
                else:
                    # Second half: random placement
                    x = self.rng.uniform(0, self.room_width - width)
                    y = self.rng.uniform(0, self.room_height - depth)
                
                item = FurnitureItem(
                    name=name,
                    width=width,
                    depth=depth,
                    height=catalog_item["height"],
                    x=x, y=y,
                    rotation=rotation,
                    movable=True
                )
                
                # Check validity
                item_rect = item.get_rect()
                
                # Don't overlap with door area
                if rects_overlap(item_rect, self.door_clear_rect):
                    continue
                
                # Don't overlap with existing items
                overlaps = False
                for existing_item in self.items:
                    existing_rect = existing_item.get_rect()
                    # Add some buffer space
                    buffered_rect = (existing_rect[0] - 10, existing_rect[1] - 10,
                                   existing_rect[2] + 20, existing_rect[3] + 20)
                    if rects_overlap(item_rect, buffered_rect):
                        overlaps = True
                        break
                
                if not overlaps:
                    self.items.append(item)
                    placed = True
                    break
            
            if not placed:
                print(f"Warning: Could not place {name} after 200 attempts")

    def reset(self, seed=None, options=None):
        """Reset the environment with enhanced initialization."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.current_step = 0
        self.best_reward_so_far = float('-inf')
        self.episode_violations = []
        self.exploration_bonus = 1.0
        self.state_visits = {}
        
        # Occasionally reset to a completely new layout for exploration
        if self.rng.random() < 0.1:  # 10% chance
            self.items = []
            self._create_curriculum_layout()
        
        return self._get_observation(), {}

    def step(self, action):
        """Enhanced step function with better reward shaping."""
        self.current_step += 1
        
        # Store previous state for comparison
        prev_violations = self._count_violations()
        prev_connectivity = self._check_all_connectivity()
        prev_relationships = self._calculate_relationships_score()
        
        # Apply action
        action_applied = self._apply_enhanced_action(action)
        
        # Calculate reward with improvement tracking
        reward = self._calculate_enhanced_reward(
            prev_violations, prev_connectivity, prev_relationships, action_applied
        )
        
        # Track violations over time
        current_violations = self._count_violations()
        self.episode_violations.append(current_violations["hard"])
        
        # Exploration bonus
        state_hash = self._get_state_hash()
        if state_hash not in self.state_visits:
            self.state_visits[state_hash] = 0
            reward += 0.1 * self.exploration_bonus  # Exploration bonus
        self.state_visits[state_hash] += 1
        
        # Decay exploration bonus
        self.exploration_bonus *= self.exploration_bonus_decay
        
        # Early termination conditions
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Success condition
        if current_violations["hard"] == 0 and self._check_all_connectivity():
            terminated = True
            reward += 10.0  # Large success bonus
            if self.current_step < self.max_steps * 0.8:
                reward += 5.0  # Early completion bonus
        
        # Failure condition (too many violations for too long)
        if (len(self.episode_violations) > 50 and 
            all(v >= 2 for v in self.episode_violations[-50:])):
            truncated = True
            reward -= 2.0  # Failure penalty
        
        info = {
            "violations_hard": current_violations["hard"],
            "violations_soft": current_violations["soft"],
            "connectivity": 1.0 if self._check_all_connectivity() else 0.0,
            "relationships": self._calculate_relationships_score(),
            "exploration_states": len(self.state_visits),
            "difficulty_stage": self.difficulty_stage,
        }
        
        return self._get_observation(), reward, terminated, truncated, info

    def _apply_enhanced_action(self, action: int) -> bool:
        """Apply enhanced action with smart movements."""
        max_items = 6  # Fixed max items for consistent action space
        if len(self.items) == 0:
            return False
            
        item_idx = action // 7
        move_type = action % 7
        
        # If action refers to a non-existent item, return False
        if item_idx >= len(self.items):
            return False
            
        item = self.items[item_idx]
        if not item.movable:
            return False
        
        # Store original state
        orig_x, orig_y, orig_rot = item.x, item.y, item.rotation
        orig_width, orig_depth = item.width, item.depth
        
        # Apply action
        action_applied = True
        
        if move_type == 0:  # Move up
            item.y = max(0, item.y - self.step_size)
        elif move_type == 1:  # Move down
            rect = item.get_rect()
            item.y = min(self.room_height - rect[3], item.y + self.step_size)
        elif move_type == 2:  # Move left
            item.x = max(0, item.x - self.step_size)
        elif move_type == 3:  # Move right
            rect = item.get_rect()
            item.x = min(self.room_width - rect[2], item.x + self.step_size)
        elif move_type == 4:  # Rotate
            old_rot = item.rotation
            item.rotate_90(self.allowed_rotations)
            
            if (old_rot % 180) != (item.rotation % 180):
                item.width, item.depth = item.depth, item.width
            
            # Check bounds after rotation
            rect = item.get_rect()
            if (rect[0] + rect[2] > self.room_width or 
                rect[1] + rect[3] > self.room_height):
                item.rotation = old_rot
                item.width, item.depth = orig_width, orig_depth
                action_applied = False
        elif move_type == 5:  # Smart move to center
            center_x = (self.room_width - item.width) / 2.0
            center_y = (self.room_height - item.depth) / 2.0
            
            # Move towards center gradually
            dx = center_x - item.x
            dy = center_y - item.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > self.step_size:
                item.x += (dx / distance) * self.step_size
                item.y += (dy / distance) * self.step_size
            else:
                item.x = center_x
                item.y = center_y
                
        elif move_type == 6:  # Smart move to wall
            # Move to nearest wall
            distances_to_walls = [
                item.x,  # left wall
                self.room_width - (item.x + item.width),  # right wall
                item.y,  # top wall
                self.room_height - (item.y + item.depth)  # bottom wall
            ]
            
            min_dist_idx = distances_to_walls.index(min(distances_to_walls))
            
            if min_dist_idx == 0:  # Move to left wall
                item.x = max(0, item.x - self.step_size)
            elif min_dist_idx == 1:  # Move to right wall
                target_x = self.room_width - item.width
                item.x = min(target_x, item.x + self.step_size)
            elif min_dist_idx == 2:  # Move to top wall
                item.y = max(0, item.y - self.step_size)
            else:  # Move to bottom wall
                target_y = self.room_height - item.depth
                item.y = min(target_y, item.y + self.step_size)
        
        # Check if new position is valid
        if self._item_has_overlap(item):
            item.x, item.y, item.rotation = orig_x, orig_y, orig_rot
            item.width, item.depth = orig_width, orig_depth
            action_applied = False
        
        return action_applied

    def _get_state_hash(self) -> str:
        """Get a hash representing the current state for exploration tracking."""
        state_data = []
        for item in self.items:
            x, y, _, _ = item.get_rect()
            # Round to grid positions to reduce state space
            grid_x = round(x / 20) * 20
            grid_y = round(y / 20) * 20
            state_data.append((item.name, grid_x, grid_y, item.rotation))
        return str(sorted(state_data))

    def _get_observation(self) -> np.ndarray:
        """Get enhanced observation with fixed size for all curriculum stages."""
        max_items = 6
        obs = []
        
        # Add item features (pad with zeros for missing items)
        for i in range(max_items):
            if i < len(self.items):
                item = self.items[i]
                x, y, w, h = item.get_rect()
                center_x, center_y = item.get_center()
                
                obs.extend([
                    x / self.room_width,
                    y / self.room_height,
                    w / self.room_width,
                    h / self.room_height,
                    item.rotation / 360.0,
                    # Distance to room center
                    abs(center_x - self.room_width/2) / (self.room_width/2),
                ])
            else:
                # Pad with zeros for non-existent items
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Global features (enhanced)
        violations = self._count_violations()
        connectivity = self._check_all_connectivity()
        relationships = self._calculate_relationships_score()
        
        obs.extend([
            1.0 if connectivity else 0.0,
            max(0, 1.0 - violations["hard"] / 5.0),  # Normalized hard violations
            max(0, 1.0 - violations["soft"] / 10.0),  # Normalized soft violations
            relationships,
            self.current_step / self.max_steps,  # Progress through episode
            1.0 if self._has_turning_space() else 0.0,
            self.difficulty_stage / self.max_difficulty,  # Current difficulty
            min(1.0, len(self.state_visits) / 100.0)  # Exploration progress
        ])
        
        return np.array(obs, dtype=np.float32)

    def _calculate_enhanced_reward(self, prev_violations: Dict, prev_connectivity: bool, 
                                 prev_relationships: float, action_applied: bool) -> float:
        """Calculate reward with better shaping and progress tracking."""
        reward = 0.0
        
        # Base reward structure
        current_violations = self._count_violations()
        current_connectivity = self._check_all_connectivity()
        current_relationships = self._calculate_relationships_score()
        
        # Reward for reducing violations (most important)
        violation_change = prev_violations["hard"] - current_violations["hard"]
        reward += violation_change * 3.0  # Strong reward for reducing hard violations
        
        soft_violation_change = prev_violations["soft"] - current_violations["soft"]
        reward += soft_violation_change * 0.5  # Moderate reward for soft violations
        
        # Reward for connectivity improvements
        if current_connectivity and not prev_connectivity:
            reward += 2.0  # Connectivity achieved
        elif not current_connectivity and prev_connectivity:
            reward -= 1.0  # Connectivity lost
        
        # Reward for relationship improvements
        relationship_change = current_relationships - prev_relationships
        reward += relationship_change * 1.0
        
        # Penalty structure
        reward -= current_violations["hard"] * 0.5  # Ongoing penalty for violations
        reward -= current_violations["soft"] * 0.1
        
        # Small time penalty to encourage efficiency
        reward -= 0.005
        
        # Penalty for ineffective actions
        if not action_applied:
            reward -= 0.02
        
        # Bonus rewards for good states
        if current_violations["hard"] == 0:
            reward += 0.5  # No hard violations bonus
            
            if current_connectivity:
                reward += 0.3  # Connectivity bonus
                
                if current_relationships > 0.8:
                    reward += 0.2  # Good relationships bonus
                    
                    if self._has_turning_space():
                        reward += 0.1  # Turning space bonus
        
        return reward

    def _item_has_overlap(self, target_item: FurnitureItem) -> bool:
        """Check if an item overlaps with others or boundaries."""
        target_rect = target_item.get_rect()
        
        # Check room boundaries
        if (target_rect[0] < 0 or target_rect[1] < 0 or
            target_rect[0] + target_rect[2] > self.room_width or
            target_rect[1] + target_rect[3] > self.room_height):
            return True
        
        # Check door clear area
        if rects_overlap(target_rect, self.door_clear_rect):
            return True
        
        # Check other items
        for item in self.items:
            if item is target_item:
                continue
            if rects_overlap(target_rect, item.get_rect()):
                return True
        
        return False

    def _count_violations(self) -> Dict[str, int]:
        """Count hard and soft violations."""
        hard_violations = 0
        soft_violations = 0
        
        # Check overlaps (hard violation)
        for i, item1 in enumerate(self.items):
            for j, item2 in enumerate(self.items[i+1:], i+1):
                if rects_overlap(item1.get_rect(), item2.get_rect()):
                    hard_violations += 1
        
        # Check door clearance (hard violation)
        for item in self.items:
            if rects_overlap(item.get_rect(), self.door_clear_rect):
                hard_violations += 1
        
        # Check individual clearances (soft violation)
        for item in self.items:
            if not self._item_has_adequate_clearance(item):
                soft_violations += 1
        
        return {"hard": hard_violations, "soft": soft_violations}

    def _item_has_adequate_clearance(self, item: FurnitureItem) -> bool:
        """Check if item has adequate clearance."""
        clearance_spec = self.clearance_specs.get(item.name, {})
        required_front = clearance_spec.get("front_clearance", self.general_clearance)
        
        if required_front <= 0:
            return True
        
        # Check front clearance (assume +y direction is front)
        x, y, w, h = item.get_rect()
        clearance_rect = (x, y + h, w, required_front)
        
        # Check room bounds
        if clearance_rect[1] + clearance_rect[3] > self.room_height:
            return False
        
        # Check overlaps with other items
        for other_item in self.items:
            if other_item is item:
                continue
            if rects_overlap(clearance_rect, other_item.get_rect()):
                return False
        
        return True

    def _has_turning_space(self) -> bool:
        """Check if there's adequate turning space."""
        diameter = self.turning_diameter
        radius = diameter / 2.0
        step = 40.0  # Check every 40cm for efficiency
        
        for x in np.arange(radius, self.room_width - radius, step):
            for y in np.arange(radius, self.room_height - radius, step):
                circle_rect = (x - radius, y - radius, diameter, diameter)
                
                if rects_overlap(circle_rect, self.door_clear_rect):
                    continue
                
                blocked = False
                for item in self.items:
                    if rects_overlap(circle_rect, item.get_rect()):
                        blocked = True
                        break
                
                if not blocked:
                    return True
        
        return False

    def _check_all_connectivity(self) -> bool:
        """Check if all required paths are clear."""
        door_center = (self.door_rect[0] + self.door_rect[2]/2, 
                      self.door_rect[1] + self.door_rect[3]/2)
        
        major_items = [item for item in self.items 
                      if item.name in ["Bed", "Sofa", "Study Table", "Wardrobe"]]
        
        for item in major_items:
            if not self._path_exists(door_center, item.get_center()):
                return False
        
        return True

    def _path_exists(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """Enhanced pathfinding with better grid resolution."""
        cell_size = 15.0  # Smaller cells for better accuracy
        grid_w = int(math.ceil(self.room_width / cell_size))
        grid_h = int(math.ceil(self.room_height / cell_size))
        
        # Mark occupied cells
        occupied = np.zeros((grid_h, grid_w), dtype=bool)
        
        padding = self.path_width / 2.0
        obstacles = [item.get_rect() for item in self.items] + [self.door_clear_rect]
        
        for (x, y, w, h) in obstacles:
            # Expand obstacle by padding
            x1 = max(0, int((x - padding) / cell_size))
            y1 = max(0, int((y - padding) / cell_size))
            x2 = min(grid_w - 1, int((x + w + padding) / cell_size))
            y2 = min(grid_h - 1, int((y + h + padding) / cell_size))
            occupied[y1:y2+1, x1:x2+1] = True
        
        # Convert world coordinates to grid coordinates
        start_grid = (int(start[0] / cell_size), int(start[1] / cell_size))
        end_grid = (int(end[0] / cell_size), int(end[1] / cell_size))
        
        # Bounds check
        if (not (0 <= start_grid[0] < grid_w and 0 <= start_grid[1] < grid_h) or
            not (0 <= end_grid[0] < grid_w and 0 <= end_grid[1] < grid_h)):
            return False
        
        # BFS pathfinding
        queue = deque([start_grid])
        visited = {start_grid}
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) == end_grid:
                return True
            
            # Check 8-connected neighbors for better pathfinding
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < grid_w and 0 <= ny < grid_h and
                    not occupied[ny, nx] and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return False

    def _calculate_relationships_score(self) -> float:
        """Calculate how well furniture relationships are satisfied."""
        if not self.functional_pairs:
            return 1.0
        
        total_score = 0.0
        total_checks = 0
        
        name_to_item = {item.name: item for item in self.items}
        
        for anchor_name, spec in self.functional_pairs.items():
            anchor_item = name_to_item.get(anchor_name)
            if not anchor_item:
                continue
            
            required_partners = spec.get("required_partners", [])
            max_distance = spec.get("max_distance", 80.0)
            
            for partner_name in required_partners:
                partner_item = name_to_item.get(partner_name)
                if not partner_item:
                    continue
                
                total_checks += 1
                distance = euclidean_distance(anchor_item.get_center(), 
                                            partner_item.get_center())
                
                if distance <= max_distance:
                    total_score += 1.0
                else:
                    # Improved partial credit with smoother decay
                    partial = max(0, 1.0 - (distance - max_distance) / (max_distance * 2))
                    total_score += partial
        
        return total_score / max(total_checks, 1)


# =========================
# Enhanced Training Functions
# =========================
def parse_planner_layout(planner_data: Dict, units_scale: float = 1.0):
    """Parse planner layout data."""
    try:
        room_width = float(planner_data["room"]["width"]) * units_scale
        room_height = float(planner_data["room"]["height"]) * units_scale
        
        # Handle door - default if not specified
        door_width = 90.0
        door_x = (room_width - door_width) / 2.0
        door_rect = [door_x, 0.0, door_width, 5.0]
        
        # Parse openings if they exist
        for opening in planner_data.get("openings", []):
            if opening.get("type", "").lower() == "door":
                door_x = float(opening.get("x", door_x)) * units_scale
                door_y = float(opening.get("y", 0.0)) * units_scale
                door_width = float(opening.get("width", door_width)) * units_scale
                door_height = float(opening.get("height", 5.0)) * units_scale
                door_rect = [door_x, door_y, door_width, door_height]
                break
        
        # Parse furniture
        items = []
        for furn in planner_data.get("furniture", []):
            items.append({
                "name": furn.get("name", ""),
                "x": float(furn.get("x", 0)) * units_scale,
                "y": float(furn.get("y", 0)) * units_scale,
                "width": float(furn.get("width", 50)) * units_scale,
                "depth": float(furn.get("height", 50)) * units_scale,  # height->depth mapping
                "rotation": int(furn.get("rotation", 0))
            })
        
        room_layout = {
            "room": {"width": room_width, "height": room_height},
            "door_rect": door_rect
        }
        
        return room_layout, items
        
    except Exception as e:
        print(f"Error parsing planner layout: {e}")
        return None, None


def create_enhanced_environment(constraints_path: str, 
                              planner_path: Optional[str] = None,
                              units_scale: float = 1.0,
                              seed: int = 42,
                              difficulty_stage: int = 0) -> EnhancedBarrierFreeEnv:
    """Create the enhanced training environment."""
    
    # Load constraints
    with open(constraints_path, 'r') as f:
        constraints = json.load(f)
    
    # Load planner layout if provided
    room_layout = None
    initial_items = None
    
    if planner_path and os.path.exists(planner_path):
        with open(planner_path, 'r') as f:
            planner_data = json.load(f)
        room_layout, initial_items = parse_planner_layout(planner_data, units_scale)
    
    # Create environment
    env = EnhancedBarrierFreeEnv(
        constraints=constraints,
        room_layout=room_layout,
        seed=seed,
        max_steps=CONFIG["MAX_STEPS_PER_EPISODE"],
        initial_items=initial_items,
        difficulty_stage=difficulty_stage
    )
    
    return env


class EnhancedVecEnvWrapper:
    """Wrapper to enable curriculum learning in vectorized environments."""
    
    def __init__(self, env):
        self.env = env
        self.current_difficulty = 0
    
    def set_difficulty(self, stage: int):
        """Set difficulty stage for curriculum learning."""
        self.current_difficulty = stage
        # Update underlying environment
        if hasattr(self.env, 'envs'):
            for env_instance in self.env.envs:
                if hasattr(env_instance, 'env'):
                    env_instance.env.set_difficulty(stage)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped environment."""
        return getattr(self.env, name)


# =========================
# Main Enhanced Training Function
# =========================
def main():
    print("Starting Enhanced RL Training with Curriculum Learning")
    print("=" * 60)
    
    # Get file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    constraints_path = os.path.join(script_dir, CONFIG["CONSTRAINTS_FILE"])
    planner_path = os.path.join(script_dir, CONFIG["PLANNER_LAYOUT_FILE"])
    
    # Validate files
    if not os.path.exists(constraints_path):
        raise FileNotFoundError(f"Constraints file not found: {constraints_path}")
    
    if not os.path.exists(planner_path):
        print(f"Warning: Planner file not found: {planner_path}")
        print("Will use curriculum-based initialization")
        planner_path = None
    
    # Create environment
    print("Creating enhanced environment...")
    env = create_enhanced_environment(
        constraints_path=constraints_path,
        planner_path=planner_path,
        units_scale=CONFIG["UNITS_SCALE"],
        seed=CONFIG["SEED"],
        difficulty_stage=0  # Start with easiest stage
    )
    
    print(f"Environment created with {len(env.items)} furniture items")
    print(f"Room dimensions: {env.room_width:.0f} x {env.room_height:.0f} cm")
    print(f"Action space: {env.action_space.n} actions")
    print(f"Observation space: {env.observation_space.shape}")
    
    # Validate environment
    try:
        check_env(env, warn=True)
        print("Environment validation passed")
    except Exception as e:
        print(f"Environment validation warning: {e}")
    
    # Wrap environment for stable-baselines3
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)
    
    env = Monitor(env, "./logs", allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    
    # Create wrapper for curriculum learning
    env_wrapper = EnhancedVecEnvWrapper(env)
    
    # Create PPO model with enhanced hyperparameters
    print("Creating enhanced PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lambda f: f * 3e-4,  # Learning rate annealing
        n_steps=4096,  # Larger batch for better learning
        batch_size=128,
        n_epochs=15,  # More epochs for better policy updates
        gamma=0.995,  # Slightly higher discount factor
        gae_lambda=0.98,
        clip_range=lambda f: f * 0.2,  # Adaptive clipping
        ent_coef=0.02,  # Higher entropy for more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=CONFIG["SEED"],
        device="cpu",  # Force CPU to avoid GPU warning
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=dict(
            net_arch=[256, 256, 128],  # Larger network
            activation_fn=lambda: __import__('torch.nn', fromlist=['ReLU']).ReLU()
        )
    )
    
    # Setup curriculum learning callback
    curriculum_callback = CurriculumCallback(
        env_wrapper=env_wrapper,
        total_timesteps=CONFIG["TOTAL_TIMESTEPS"],
        stages=CONFIG["CURRICULUM_STAGES"],
        verbose=1
    )
    
    # Train the model with curriculum learning
    print(f"Starting enhanced training for {CONFIG['TOTAL_TIMESTEPS']} timesteps...")
    print(f"Curriculum stages: {CONFIG['CURRICULUM_STAGES']}")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=CONFIG["TOTAL_TIMESTEPS"],
            progress_bar=True,
            callback=curriculum_callback
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    # Save model
    save_path = os.path.join(script_dir, CONFIG["SAVE_PATH"])
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    
    # Run enhanced evaluation
    if CONFIG["RUN_EVAL"]:
        print("\nRunning enhanced evaluation...")
        
        # Get the underlying environment and set to highest difficulty
        eval_env = env.envs[0].env  # Unwrap from Monitor and DummyVecEnv
        eval_env.set_difficulty(CONFIG["CURRICULUM_STAGES"] - 1)  # Highest difficulty
        
        # Run multiple evaluation episodes
        num_eval_episodes = 3
        all_rewards = []
        all_results = []
        
        for episode in range(num_eval_episodes):
            print(f"\n--- Evaluation Episode {episode + 1} ---")
            
            # Reset environment
            obs, _ = eval_env.reset(seed=CONFIG["SEED"] + episode)
            total_reward = 0.0
            step_count = 0
            
            # Initial state
            violations = eval_env._count_violations()
            connectivity = eval_env._check_all_connectivity()
            relationships = eval_env._calculate_relationships_score()
            
            print(f"Initial state:")
            print(f"  Hard violations: {violations['hard']}")
            print(f"  Soft violations: {violations['soft']}")
            print(f"  Connectivity: {connectivity}")
            print(f"  Relationships: {relationships:.3f}")
            
            best_reward = float('-inf')
            no_improvement_count = 0
            
            # Run evaluation steps
            for step in range(CONFIG["EVAL_STEPS"]):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                step_count += 1
                
                # Track improvement
                if reward > best_reward:
                    best_reward = reward
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Print progress
                if (step + 1) % 100 == 0:
                    print(f"  Step {step + 1}: Reward = {reward:.3f}, "
                          f"Hard violations = {info['violations_hard']}, "
                          f"Connectivity = {info['connectivity']:.1f}")
                
                # Early termination conditions
                if terminated:
                    print(f"Episode terminated successfully at step {step_count}")
                    break
                elif truncated:
                    print(f"Episode truncated at step {step_count}")
                    break
                elif no_improvement_count > 100:
                    print(f"No improvement for 100 steps, ending episode at step {step_count}")
                    break
            
            # Episode results
            episode_result = {
                "episode": episode + 1,
                "total_reward": total_reward,
                "steps": step_count,
                "avg_reward": total_reward / step_count,
                "final_hard_violations": info.get('violations_hard', 0),
                "final_soft_violations": info.get('violations_soft', 0),
                "final_connectivity": info.get('connectivity', 0),
                "final_relationships": info.get('relationships', 0),
                "exploration_states": info.get('exploration_states', 0),
                "success": info.get('violations_hard', 1) == 0 and info.get('connectivity', 0) == 1.0
            }
            
            all_rewards.append(total_reward)
            all_results.append(episode_result)
            
            print(f"Episode {episode + 1} Results:")
            for key, value in episode_result.items():
                if key != "episode":
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
        
        # Overall evaluation summary
        print(f"\n{'='*50}")
        print("OVERALL EVALUATION SUMMARY")
        print(f"{'='*50}")
        
        avg_reward = sum(all_rewards) / len(all_rewards)
        success_rate = sum(1 for r in all_results if r['success']) / len(all_results)
        avg_violations = sum(r['final_hard_violations'] for r in all_results) / len(all_results)
        avg_connectivity = sum(r['final_connectivity'] for r in all_results) / len(all_results)
        avg_relationships = sum(r['final_relationships'] for r in all_results) / len(all_results)
        
        print(f"Average total reward: {avg_reward:.2f}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Average hard violations: {avg_violations:.1f}")
        print(f"Average connectivity: {avg_connectivity:.3f}")
        print(f"Average relationships: {avg_relationships:.3f}")
        
        # Save the best layout from evaluation
        best_episode = max(all_results, key=lambda x: x['total_reward'])
        print(f"\nBest episode: {best_episode['episode']} (reward: {best_episode['total_reward']:.2f})")
        
        # Save optimized layout from the final episode
        furniture_data = []
        for item in eval_env.items:
            rect = item.get_rect()
            furniture_data.append({
                "name": item.name,
                "x": rect[0],
                "y": rect[1], 
                "width": rect[2],
                "height": rect[3],
                "zHeight": str(item.height),
                "rotation": item.rotation
            })
        
        optimized_layout = {
            "room": {
                "width": eval_env.room_width,
                "height": eval_env.room_height
            },
            "furniture": furniture_data,
            "openings": [],
            "evaluation_summary": {
                "num_episodes": num_eval_episodes,
                "average_reward": avg_reward,
                "success_rate": success_rate,
                "average_hard_violations": avg_violations,
                "average_connectivity": avg_connectivity,
                "average_relationships": avg_relationships,
                "best_episode_reward": best_episode['total_reward'],
                "training_time_minutes": training_time / 60
            },
            "individual_episodes": all_results
        }
        
        output_path = os.path.join(script_dir, CONFIG["OPTIMIZED_LAYOUT_OUT"])
        with open(output_path, 'w') as f:
            json.dump(optimized_layout, f, indent=2)
        
        print(f"Optimized layout saved to {output_path}")
    
    print(f"\n{'='*60}")
    print("ENHANCED TRAINING AND EVALUATION COMPLETE!")
    print(f"{'='*60}")
    
    return model, env


if __name__ == "__main__":
    try:
        model, env = main()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()