import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

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
        self.move_step = 5  # Reduced for finer adjustments
        
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
        
        # Ensure no initial overlaps
        self._resolve_initial_overlaps()
        
        return self._get_observation(), {}
    
    def _resolve_initial_overlaps(self):
        """Adjust initial positions to resolve overlaps"""
        max_iterations = 100
        for _ in range(max_iterations):
            overlaps = False
            for i, item1 in enumerate(self.furniture_items):
                for j, item2 in enumerate(self.furniture_items[i+1:], i+1):
                    if self._items_overlap(item1, item2):
                        overlaps = True
                        # Move item2 slightly to avoid overlap
                        item2.x += random.uniform(10, 20)
                        item2.y += random.uniform(10, 20)
                        item2.x = np.clip(item2.x, 0, self.room_width - item2.width)
                        item2.y = np.clip(item2.y, 0, self.room_height - item2.height)
            if not overlaps:
                break
    
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
    
    def _items_overlap(self, item1: FurnitureItem, item2: FurnitureItem) -> bool:
        """Check if two furniture items overlap"""
        x1, y1, w1, h1 = item1.x, item1.y, item1.width, item1.height
        x2, y2, w2, h2 = item2.x, item2.y, item2.width, item2.height
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    def _count_violations(self) -> int:
        """Count accessibility and layout violations"""
        violations = 0
        violation_details = []

        # Check clearances
        for item in self.furniture_items:
            clearance = self.constraints.furniture_clearances.get(item.name, {})
            required_front = clearance.get("front_clearance", self.constraints.general_clearance)
            required_side = clearance.get("left_access", self.constraints.general_clearance)

            # Front clearance
            front_clear = min(self.room_height - (item.y + item.height), item.y) if item.rotation in [0, 180] else min(self.room_width - (item.x + item.width), item.x)
            if front_clear < required_front:
                violations += 1
                violation_details.append(f"{item.name}: Insufficient front clearance ({front_clear}cm < {required_front}cm)")

            # Side clearances
            left_clear = item.x if item.rotation in [0, 180] else item.y
            right_clear = (self.room_width - (item.x + item.width)) if item.rotation in [0, 180] else (self.room_height - (item.y + item.height))
            if min(left_clear, right_clear) < required_side:
                violations += 1
                violation_details.append(f"{item.name}: Insufficient side clearance ({min(left_clear, right_clear)}cm < {required_side}cm)")

        # Check path width and turning radius
        for i, item1 in enumerate(self.furniture_items):
            for j, item2 in enumerate(self.furniture_items[i+1:], i+1):
                if self._items_overlap(item1, item2):
                    violations += 1
                    violation_details.append(f"Overlap between {item1.name} and {item2.name}")
                dist = np.hypot(item1.x - item2.x, item1.y - item2.y)
                if dist < self.constraints.path_width:
                    violations += 1
                    violation_details.append(f"Path violation between {item1.name} and {item2.name} ({dist}cm < {self.constraints.path_width}cm)")

        # Check functional pairs
        for pair, rules in self.constraints.furniture_relationships.get("functional_pairs", {}).items():
            item1_name, item2_name = pair.split("-")
            item1 = next((i for i in self.furniture_items if i.name == item1_name), None)
            item2 = next((i for i in self.furniture_items if i.name == item2_name), None)
            if item1 and item2:
                dist = np.hypot(item1.x - item2.x, item1.y - item2.y)
                if dist > rules.get("max_distance", float('inf')) or dist < rules.get("min_distance", 0):
                    violations += 1
                    violation_details.append(f"{pair} distance violation ({dist}cm outside [{rules.get('min_distance', 0)}-{rules.get('max_distance', float('inf'))}cm])")

        # Debug: Log violation details
        if violation_details:
            logger.debug("Violation details: %s", violation_details)

        return violations

    def _calculate_accessibility_score(self) -> float:
        """Calculate accessibility score based on violations and clearances"""
        violations = self._count_violations()
        max_violations = 20  # Arbitrary max for normalization
        score = max(0, 1 - (violations / max_violations))
        return score

    def _calculate_layout_efficiency(self) -> float:
        """Calculate layout efficiency based on space usage"""
        total_area = self.room_width * self.room_height
        furniture_area = sum(item.width * item.height for item in self.furniture_items)
        efficiency = min(1.0, furniture_area / total_area * 1.5)  # Cap at 1.0, scale by 1.5 for realistic use
        return efficiency

    def _calculate_layout_quality(self) -> float:
        """Calculate layout quality reward"""
        violations = self._count_violations()
        accessibility_score = self._calculate_accessibility_score()
        efficiency = self._calculate_layout_efficiency()

        # Base reward: penalize violations, reward accessibility and efficiency
        base_reward = -violations * 1.0 + accessibility_score * 5.0 + efficiency * 5.0

        # Bonus for functional pairs
        pair_bonus = 0
        for pair, rules in self.constraints.furniture_relationships.get("functional_pairs", {}).items():
            item1_name, item2_name = pair.split("-")
            item1 = next((i for i in self.furniture_items if i.name == item1_name), None)
            item2 = next((i for i in self.furniture_items if i.name == item2_name), None)
            if item1 and item2:
                dist = np.hypot(item1.x - item2.x, item1.y - item2.y)
                if rules.get("min_distance", 0) <= dist <= rules.get("max_distance", float('inf')):
                    pair_bonus += 2.0

        # Clip total reward to prevent extreme values
        total_reward = np.clip(base_reward + pair_bonus, -5, 5)
        return total_reward

    def step(self, action):
        """Execute one step in the environment"""
        furniture_idx, action_type, direction = action
        furniture_idx = int(furniture_idx)
        action_type = int(action_type)
        direction = int(direction)
        
        item = self.furniture_items[furniture_idx]
        old_x, old_y, old_rot = item.x, item.y, item.rotation
        
        if action_type == 0:  # Move
            if direction == 0:    # Up
                item.y -= self.move_step
            elif direction == 1:  # Down
                item.y += self.move_step
            elif direction == 2:  # Left
                item.x -= self.move_step
            elif direction == 3:  # Right
                item.x += self.move_step
        elif action_type == 1:  # Rotate
            item.rotation = (direction * 90) % 360
            if direction in [1, 3]:
                item.width, item.height = item.height, item.width
        
        # Check boundaries
        item.x = np.clip(item.x, 0, self.room_width - item.width)
        item.y = np.clip(item.y, 0, self.room_height - item.height)
        
        # Check for overlaps
        overlap = False
        for other_item in self.furniture_items:
            if other_item == item:
                continue
            if self._items_overlap(item, other_item):
                overlap = True
                break
        
        if overlap:
            # Revert action and mark as truncated
            item.x, item.y, item.rotation = old_x, old_y, old_rot
            reward = -10.0
            done = False
            truncated = True
        else:
            reward = self._calculate_layout_quality()
            done = self.current_step >= self.max_steps
            truncated = False
        
        self.current_step += 1
        
        total_reward = reward
        total_reward = np.clip(total_reward, -5, 5)
        
        info = {
            'violations': self._count_violations(),
            'accessibility_score': self._calculate_accessibility_score(),
            'layout_efficiency': self._calculate_layout_efficiency(),
            'step': self.current_step
        }
        
        return self._get_observation(), total_reward, done, truncated, info

    def visualize_layout(self, save_path: Optional[str] = None):
        """Visualize the current layout"""
        fig, ax = plt.subplots(figsize=(12, 9))
        room_rect = Rectangle((0, 0), self.room_width, self.room_height, 
                            linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(room_rect)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.furniture_items)))
        for i, item in enumerate(self.furniture_items):
            rect = Rectangle((item.x, item.y), item.width, item.height,
                           linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.7)
            ax.add_patch(rect)
            ax.text(item.x + item.width/2, item.y + item.height/2, item.name,
                   ha='center', va='center', fontsize=8, weight='bold')
            
            # Highlight overlaps (should be none with new constraint)
            for j, other_item in enumerate(self.furniture_items[i+1:], i+1):
                if self._items_overlap(item, other_item):
                    overlap_rect = Rectangle((min(item.x, other_item.x), min(item.y, other_item.y)),
                                           max(item.x + item.width, other_item.x + other_item.width) - min(item.x, other_item.x),
                                           max(item.y + item.height, other_item.y + other_item.height) - min(item.y, other_item.y),
                                           linewidth=2, edgecolor='red', facecolor='none', alpha=0.5)
                    ax.add_patch(overlap_rect)
        
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

    def get_layout_json(self):
        """Return current layout as JSON-compatible dict"""
        layout = {
            "room": {"width": self.room_width, "height": self.room_height},
            "furniture": [{"name": item.name, "x": item.x, "y": item.y, "width": item.width,
                          "height": item.height, "zHeight": item.z_height, "rotation": item.rotation}
                         for item in self.furniture_items],
            "openings": self.openings,
            "accessibility_score": self._calculate_accessibility_score(),
            "layout_efficiency": self._calculate_layout_efficiency(),
            "total_violations": self._count_violations()
        }
        return layout

class CustomCallback(BaseCallback):
    """Callback to log and visualize episode metrics at the end of training"""
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.episode_rewards = []
        self.violations = []
        self.accessibility_scores = []
        self.efficiencies = []
        self.episode_count = 0
    
    def _on_step(self):
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            self.episode_rewards.append(self.locals["rewards"][0])
            self.violations.append(info["violations"])
            self.accessibility_scores.append(info["accessibility_score"])
            self.efficiencies.append(info["layout_efficiency"])
            self.episode_count += 1
        
        return True
    
    def _on_training_end(self):
        """Generate a single plot of all metrics at the end of training"""
        plt.figure(figsize=(12, 6))
        
        episodes = range(1, self.episode_count + 1)
        
        plt.plot(episodes, self.episode_rewards, label='Episode Reward', color='blue')
        plt.plot(episodes, self.violations, label='Violations', color='red')
        plt.plot(episodes, self.accessibility_scores, label='Accessibility Score', color='green')
        plt.plot(episodes, self.efficiencies, label='Efficiency', color='orange')
        
        plt.title('Model Performance Over Training')
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, 'training_performance_final.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training performance plot saved to {save_path}")

def train_barrier_free_model(room_layout_path: str, 
                           constraints_path: str,
                           total_timesteps: int = 3000000,
                           model_save_path: str = "barrier_free_model"):
    """Train the barrier-free furniture arrangement model"""
    with open(room_layout_path, 'r') as f:
        room_layout = json.load(f)
    
    with open(constraints_path, 'r') as f:
        constraints = json.load(f)
    
    env = BarrierFreeEnvironment(room_layout, constraints, accessibility_mode="wheelchair")
    
    check_env(env)
    logger.info("Environment check passed!")
    
    save_dir = os.path.join(os.path.dirname(model_save_path), "training_visuals")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,  # Lowered for stability
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,  # Reduced for conservative updates
        ent_coef=0.01,  # Adjusted for better exploration balance
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[512, 512, 256]),  # Larger network
        device="cpu"  # Explicitly using CPU
    )
    
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    
    model.learn(total_timesteps=total_timesteps, callback=CustomCallback(save_dir))
    
    model.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    return model, env

def test_trained_model(model_path: str, 
                      room_layout_path: str, 
                      constraints_path: str,
                      num_episodes: int = 100):
    """Test the trained model and generate optimized layouts"""
    with open(room_layout_path, 'r') as f:
        room_layout = json.load(f)
    
    with open(constraints_path, 'r') as f:
        constraints = json.load(f)
    
    env = BarrierFreeEnvironment(room_layout, constraints)
    
    model = PPO.load(model_path, env=env)
    
    best_layout = None
    best_score = float('-inf')
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)  # Add seed for diversity
        total_reward = 0
        
        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=False)
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
    logger.info("Starting barrier-free furniture arrangement training...")
    
    model, env = train_barrier_free_model(
        room_layout_path="room-layout-1.json",
        constraints_path="constraints/constraints.json",
        total_timesteps=10000,  
        model_save_path="Outputs/RL/barrier_free_furniture_model_v9"
    )
    
    logger.info("Training completed!")
    
    logger.info("Testing trained model...")
    
    best_layout = test_trained_model(
        model_path="Outputs/RL/barrier_free_furniture_model_v9",
        room_layout_path="room-layout-1.json", 
        constraints_path="constraints/constraints.json",
        num_episodes=100
    )
    
    if best_layout:
        logger.info("=== BARRIER-FREE LAYOUT OPTIMIZATION COMPLETE ===")
        logger.info(f"Final accessibility score: {best_layout['accessibility_score']:.3f}")
        logger.info(f"Layout efficiency: {best_layout['layout_efficiency']:.3f}")
        logger.info(f"Total violations: {best_layout['total_violations']}")
        logger.info("Optimized layout saved to: optimized_barrier_free_layout.json")
    else:
        logger.warning("No valid layout generated")