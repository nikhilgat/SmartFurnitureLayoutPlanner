# ============================================================================

LAYOUT_FILE = "room-layout-1.json"              #  room layout 
MODEL_PATH = "Outputs/RL/barrier_free_furniture_model_v9.zip"  
CONSTRAINTS_PATH = "constraints/merged_barrier_free_constraints.json"  

NUM_EPISODES = 75                              
MAX_STEPS_PER_EPISODE = 700                   
USE_DETERMINISTIC_ACTIONS = False             
OUTPUT_DIRECTORY = "Outputs/RL/inference_output"         

SAVE_VISUALIZATIONS = True                    
VERBOSE_LOGGING = True                       

RUN_DIRECTLY = True                          

# ============================================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import logging
from typing import Dict, List, Optional
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings("ignore")

# Import the environment class from your training script
try:
    from Train import BarrierFreeEnvironment, FurnitureItem, AccessibilityConstraints
    print("✓ Successfully imported environment from Train.py")
except ImportError as e:
    print(f"✗ Could not import from Train.py: {e}")
    print("Make sure Train.py is in the same directory")
    exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BarrierFreeInference:
    """Inference engine for barrier-free furniture arrangement optimization"""
    
    def __init__(self, model_path: str, constraints_path: str):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to the trained PPO model (.zip file)
            constraints_path: Path to the accessibility constraints JSON file
        """
        self.model_path = model_path
        self.constraints_path = constraints_path
        self.constraints = None
        
        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(constraints_path):
            raise FileNotFoundError(f"Constraints file not found: {constraints_path}")
        
        # Load constraints
        with open(constraints_path, 'r') as f:
            self.constraints = json.load(f)
        
        logger.info(f"Initialized inference engine")
        logger.info(f"Model: {model_path}")
        logger.info(f"Constraints: {constraints_path}")
    
    def optimize_single_layout(self, 
                              room_layout: Dict,
                              num_episodes: int = 30,
                              max_steps_per_episode: int = 500,
                              use_deterministic: bool = False,
                              save_visualizations: bool = True,
                              output_dir: str = "inference_output") -> Dict:
        """
        Optimize a single room layout using the trained model
        
        Args:
            room_layout: Dictionary containing room and furniture data
            num_episodes: Number of optimization episodes
            max_steps_per_episode: Maximum steps per episode
            use_deterministic: Use deterministic actions (less exploration)
            save_visualizations: Save layout images
            output_dir: Directory to save outputs
            
        Returns:
            Best optimized layout dictionary
        """
        logger.info("="*60)
        logger.info("STARTING LAYOUT OPTIMIZATION")
        logger.info("="*60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create environment
        env = BarrierFreeEnvironment(room_layout, self.constraints)
        
        # Load trained model
        logger.info("Loading trained model...")
        model = PPO.load(self.model_path, env=env)
        logger.info("✓ Model loaded successfully")
        
        # Log room info
        room_width = room_layout['room']['width']
        room_height = room_layout['room']['height']
        furniture_count = len(room_layout['furniture'])
        
        logger.info(f"Room dimensions: {room_width} x {room_height}")
        logger.info(f"Furniture items: {furniture_count}")
        logger.info(f"Optimization episodes: {num_episodes}")
        
        # Track optimization results
        best_layout = None
        best_score = float('-inf')
        best_episode = -1
        episode_results = []
        
        # Run optimization episodes
        for episode in range(num_episodes):
            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0
            step_count = 0
            
            # Run episode
            for step in range(max_steps_per_episode):
                # Get action from model
                action, _ = model.predict(obs, deterministic=use_deterministic)
                
                # Execute action
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count = step + 1
                
                if done or truncated:
                    break
            
            # Get final layout
            layout = env.get_layout_json()
            
            # Calculate comprehensive score (weighted combination of metrics)
            comprehensive_score = self._calculate_comprehensive_score(layout)
            
            # Store episode results
            episode_data = {
                'episode': episode + 1,
                'total_reward': episode_reward,
                'steps': step_count,
                'accessibility_score': layout['accessibility_score'],
                'layout_efficiency': layout['layout_efficiency'],
                'total_violations': layout['total_violations'],
                'comprehensive_score': comprehensive_score
            }
            episode_results.append(episode_data)
            
            # Check if this is the best layout so far
            if comprehensive_score > best_score:
                best_score = comprehensive_score
                best_layout = layout.copy()
                best_episode = episode + 1
                
                logger.info(f"🎉 New best layout found at episode {episode + 1}!")
                logger.info(f"   Comprehensive score: {comprehensive_score:.3f}")
                logger.info(f"   Accessibility: {layout['accessibility_score']:.3f}")
                logger.info(f"   Efficiency: {layout['layout_efficiency']:.3f}")
                logger.info(f"   Violations: {layout['total_violations']}")
            
            # Log progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_accessibility = np.mean([r['accessibility_score'] for r in episode_results[-10:]])
                avg_violations = np.mean([r['total_violations'] for r in episode_results[-10:]])
                
                logger.info(f"Episode {episode + 1:3d} | "
                           f"Score: {comprehensive_score:.3f} | "
                           f"Acc: {layout['accessibility_score']:.3f} | "
                           f"Eff: {layout['layout_efficiency']:.3f} | "
                           f"Viol: {layout['total_violations']} | "
                           f"Avg(10): {avg_accessibility:.3f}")
        
        logger.info("="*60)
        logger.info("OPTIMIZATION COMPLETED")
        logger.info("="*60)
        logger.info(f"Best episode: {best_episode}/{num_episodes}")
        logger.info(f"Best comprehensive score: {best_score:.3f}")
        logger.info(f"Final accessibility score: {best_layout['accessibility_score']:.3f}")
        logger.info(f"Final layout efficiency: {best_layout['layout_efficiency']:.3f}")
        logger.info(f"Final violations: {best_layout['total_violations']}")
        
        # Save results
        self._save_optimization_results(
            best_layout, episode_results, room_layout, output_dir, best_episode
        )
        
        # Create visualizations
        if save_visualizations:
            self._create_visualizations(best_layout, episode_results, env, output_dir)
        
        return best_layout
    
    def _calculate_comprehensive_score(self, layout: Dict) -> float:
        """Calculate a weighted comprehensive score for layout quality"""
        accessibility = layout['accessibility_score']
        efficiency = layout['layout_efficiency'] 
        violations = layout['total_violations']
        
        # Weighted scoring: prioritize accessibility, penalize violations
        score = (accessibility * 0.6 + efficiency * 0.3) - (violations * 0.05)
        return max(0, score)
    
    def _save_optimization_results(self, best_layout: Dict, episode_results: List[Dict], 
                                  original_layout: Dict, output_dir: str, best_episode: int):
        """Save optimization results to JSON files"""
        # Save best optimized layout
        best_layout_path = os.path.join(output_dir, "optimized_layout.json")
        with open(best_layout_path, 'w') as f:
            json.dump(best_layout, f, indent=2)
        
        # Save original layout for comparison
        original_layout_path = os.path.join(output_dir, "original_layout.json")
        with open(original_layout_path, 'w') as f:
            json.dump(original_layout, f, indent=2)
        
        # Save detailed episode results
        results_path = os.path.join(output_dir, "episode_results.json")
        with open(results_path, 'w') as f:
            json.dump(episode_results, f, indent=2)
        
        # Create summary report
        summary = {
            "optimization_summary": {
                "total_episodes": len(episode_results),
                "best_episode": best_episode,
                "best_comprehensive_score": self._calculate_comprehensive_score(best_layout),
                "improvement_metrics": self._calculate_improvements(original_layout, best_layout)
            },
            "final_metrics": {
                "accessibility_score": best_layout['accessibility_score'],
                "layout_efficiency": best_layout['layout_efficiency'],
                "total_violations": best_layout['total_violations']
            },
            "statistics": {
                "avg_accessibility": np.mean([r['accessibility_score'] for r in episode_results]),
                "max_accessibility": max(r['accessibility_score'] for r in episode_results),
                "min_violations": min(r['total_violations'] for r in episode_results),
                "avg_violations": np.mean([r['total_violations'] for r in episode_results])
            }
        }
        
        summary_path = os.path.join(output_dir, "optimization_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Results saved to: {output_dir}")
        logger.info(f"  - Optimized layout: optimized_layout.json")
        logger.info(f"  - Summary report: optimization_summary.json")
        logger.info(f"  - Episode details: episode_results.json")
    
    def _calculate_improvements(self, original: Dict, optimized: Dict) -> Dict:
        """Calculate improvement metrics between original and optimized layouts"""
        # Create temporary environments to calculate original metrics
        temp_env_orig = BarrierFreeEnvironment(original, self.constraints)
        temp_env_orig.furniture_items = temp_env_orig._create_furniture_items(original["furniture"])
        
        original_metrics = {
            'accessibility_score': temp_env_orig._calculate_accessibility_score(),
            'layout_efficiency': temp_env_orig._calculate_layout_efficiency(),
            'violations': temp_env_orig._count_violations()
        }
        
        return {
            "accessibility_improvement": optimized['accessibility_score'] - original_metrics['accessibility_score'],
            "efficiency_improvement": optimized['layout_efficiency'] - original_metrics['layout_efficiency'],
            "violations_reduction": original_metrics['violations'] - optimized['total_violations']
        }
    
    def _create_visualizations(self, best_layout: Dict, episode_results: List[Dict], 
                              env: BarrierFreeEnvironment, output_dir: str):
        """Create and save visualization plots"""
        # Create layout visualization
        self._visualize_layout(best_layout, env, output_dir)
        
        # Create optimization progress plots
        self._plot_optimization_progress(episode_results, output_dir)
        
        logger.info("✓ Visualizations created")
    
    def _visualize_layout(self, layout: Dict, env: BarrierFreeEnvironment, output_dir: str):
        """Create and save layout visualization"""
        # Update environment with best layout
        env.furniture_items = env._create_furniture_items(layout["furniture"])
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Draw room
        room_rect = Rectangle((0, 0), env.room_width, env.room_height, 
                            linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.2)
        ax.add_patch(room_rect)
        
        # Draw furniture with distinct colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(env.furniture_items)))
        
        for i, item in enumerate(env.furniture_items):
            # Draw furniture piece
            rect = Rectangle((item.x, item.y), item.width, item.height,
                           linewidth=2, edgecolor='black', facecolor=colors[i], alpha=0.8)
            ax.add_patch(rect)
            
            # Add furniture label
            ax.text(item.x + item.width/2, item.y + item.height/2, 
                   item.name, ha='center', va='center', 
                   fontsize=9, weight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Draw openings (doors and windows)
        for opening in env.openings:
            wall = opening.get("wall")
            position = opening.get("position", 0)
            size = opening.get("size", 90)
            thickness = 15
            
            if wall == "bottom":
                x, y = position, 0
                width, height = size, thickness
            elif wall == "top":
                x, y = position, env.room_height - thickness
                width, height = size, thickness
            elif wall == "left":
                x, y = 0, position
                width, height = thickness, size
            elif wall == "right":
                x, y = env.room_width - thickness, position
                width, height = thickness, size
            else:
                continue
                
            opening_color = 'red' if opening.get('type') == 'door' else 'blue'
            opening_rect = Rectangle((x, y), width, height,
                                   linewidth=2, edgecolor=opening_color, 
                                   facecolor=opening_color, alpha=0.6)
            ax.add_patch(opening_rect)
        
        # Set plot properties
        ax.set_xlim(-50, env.room_width + 50)
        ax.set_ylim(-50, env.room_height + 50)
        ax.set_aspect('equal')
        
        # Add metrics to title
        title = (f'Optimized Barrier-Free Layout\n'
                f'Accessibility: {layout["accessibility_score"]:.3f} | '
                f'Efficiency: {layout["layout_efficiency"]:.3f} | '
                f'Violations: {layout["total_violations"]}')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Width (cm)', fontsize=12)
        ax.set_ylabel('Height (cm)', fontsize=12)
        
        # Add legend for openings
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=4, alpha=0.6, label='Door'),
            Line2D([0], [0], color='blue', lw=4, alpha=0.6, label='Window')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        # Save plot
        layout_path = os.path.join(output_dir, "optimized_layout.png")
        plt.savefig(layout_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_optimization_progress(self, episode_results: List[Dict], output_dir: str):
        """Create optimization progress plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        episodes = [r['episode'] for r in episode_results]
        accessibility = [r['accessibility_score'] for r in episode_results]
        efficiency = [r['layout_efficiency'] for r in episode_results]
        violations = [r['total_violations'] for r in episode_results]
        comprehensive = [r['comprehensive_score'] for r in episode_results]
        
        # Accessibility score progress
        ax1.plot(episodes, accessibility, 'g-', linewidth=2, alpha=0.8)
        ax1.fill_between(episodes, accessibility, alpha=0.3, color='green')
        ax1.set_title('Accessibility Score Progress', fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Accessibility Score')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=max(accessibility), color='green', linestyle='--', alpha=0.5)
        
        # Layout efficiency progress
        ax2.plot(episodes, efficiency, 'b-', linewidth=2, alpha=0.8)
        ax2.fill_between(episodes, efficiency, alpha=0.3, color='blue')
        ax2.set_title('Layout Efficiency Progress', fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Layout Efficiency')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=max(efficiency), color='blue', linestyle='--', alpha=0.5)
        
        # Violations over episodes
        ax3.plot(episodes, violations, 'r-', linewidth=2, alpha=0.8)
        ax3.fill_between(episodes, violations, alpha=0.3, color='red')
        ax3.set_title('Violations Over Episodes', fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Number of Violations')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=min(violations), color='red', linestyle='--', alpha=0.5)
        
        # Comprehensive score
        ax4.plot(episodes, comprehensive, 'purple', linewidth=2, alpha=0.8)
        ax4.fill_between(episodes, comprehensive, alpha=0.3, color='purple')
        ax4.set_title('Comprehensive Score Progress', fontweight='bold')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Comprehensive Score')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=max(comprehensive), color='purple', linestyle='--', alpha=0.5)
        
        plt.suptitle('Optimization Progress Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        progress_path = os.path.join(output_dir, "optimization_progress.png")
        plt.savefig(progress_path, dpi=150, bbox_inches='tight')
        plt.close()


def run_direct_mode():
    """
    🚀 DIRECT RUN MODE - Uses the configuration settings defined at the top
    This allows you to just press F5 in your IDE without command line arguments
    """
    print("🚀 Running in DIRECT MODE using configuration settings")
    print("="*60)
    
    try:
        # Validate files exist
        if not os.path.exists(LAYOUT_FILE):
            print(f"❌ Layout file not found: {LAYOUT_FILE}")
            print("💡 Update LAYOUT_FILE at the top of this script")
            return 1
            
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found: {MODEL_PATH}")
            print("💡 Update MODEL_PATH at the top of this script")
            return 1
            
        if not os.path.exists(CONSTRAINTS_PATH):
            print(f"❌ Constraints file not found: {CONSTRAINTS_PATH}")
            print("💡 Update CONSTRAINTS_PATH at the top of this script")
            return 1
        
        # Load room layout
        with open(LAYOUT_FILE, 'r') as f:
            room_layout = json.load(f)
        
        # Validate room layout structure
        if 'room' not in room_layout or 'furniture' not in room_layout:
            print("❌ Invalid room layout format. Must contain 'room' and 'furniture' keys.")
            return 1
        
        print(f"✓ Layout file: {LAYOUT_FILE}")
        print(f"✓ Model: {MODEL_PATH}")
        print(f"✓ Constraints: {CONSTRAINTS_PATH}")
        print(f"✓ Episodes: {NUM_EPISODES}")
        print(f"✓ Room size: {room_layout['room']['width']} x {room_layout['room']['height']}")
        print(f"✓ Furniture items: {len(room_layout['furniture'])}")
        print()
        
        # Initialize inference engine
        inference_engine = BarrierFreeInference(MODEL_PATH, CONSTRAINTS_PATH)
        
        # Run optimization with configured settings
        best_layout = inference_engine.optimize_single_layout(
            room_layout=room_layout,
            num_episodes=NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS_PER_EPISODE,
            use_deterministic=USE_DETERMINISTIC_ACTIONS,
            save_visualizations=SAVE_VISUALIZATIONS,
            output_dir=OUTPUT_DIRECTORY
        )
        
        print("\n" + "="*60)
        print("🎉 OPTIMIZATION COMPLETE!")
        print("="*60)
        print(f"✓ Best accessibility score: {best_layout['accessibility_score']:.3f}")
        print(f"✓ Layout efficiency: {best_layout['layout_efficiency']:.3f}")
        print(f"✓ Total violations: {best_layout['total_violations']}")
        print(f"✓ Results saved to: {OUTPUT_DIRECTORY}/")
        print(f"✓ Optimized layout: {OUTPUT_DIRECTORY}/optimized_layout.json")
        
        if SAVE_VISUALIZATIONS:
            print(f"✓ Layout visualization: {OUTPUT_DIRECTORY}/optimized_layout.png")
            print(f"✓ Progress charts: {OUTPUT_DIRECTORY}/optimization_progress.png")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        print("💡 Check the file paths in the configuration section at the top")
        return 1
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        if VERBOSE_LOGGING:
            logger.exception("Detailed error information:")
        return 1


def find_layout_file() -> Optional[str]:
    """Auto-detect a layout JSON file in the current directory"""
    # Look for common layout file patterns
    patterns = [
        'room_layout.json',
        'room-layout.json', 
        'layout.json',
        'room_layout_*.json',
        'room-layout-*.json'
    ]
    
    import glob
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]  # Return first match
    
    # Look for any JSON file that might be a layout
    json_files = glob.glob('*.json')
    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                # Check if it looks like a room layout
                if 'room' in data and 'furniture' in data:
                    return file
        except:
            continue
    
    return None

def main():
    """Main function - supports both direct mode and command-line mode"""
    
    # 🚀 DIRECT RUN MODE - Check if we should run directly with config settings
    if RUN_DIRECTLY:
        return run_direct_mode()
    
    # 🖥️ COMMAND LINE MODE - Parse arguments and run normally
    parser = argparse.ArgumentParser(
        description='Barrier-Free Furniture Layout Optimization Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplest usage - auto-detects layout file
  python inference.py
  
  # Specify layout file
  python inference.py --layout room_layout.json
  
  # Full specification
  python inference.py --model models/trained_model --constraints constraints.json --layout room_layout.json
  
  # More episodes for better optimization
  python inference.py --episodes 50
  
  # Deterministic mode (less exploration, more exploitation)
  python inference.py --deterministic
        """
    )
    
    parser.add_argument('--layout', 
                       help='Path to room layout JSON file (auto-detected if not specified)')
    parser.add_argument('--model', 
                       default='Outputs/RL/barrier_free_furniture_model_v9',
                       help='Path to trained PPO model (default: Outputs/RL/barrier_free_furniture_model_v9)')
    parser.add_argument('--constraints', 
                       default='constraints/merged_barrier_free_constraints.json',
                       help='Path to constraints JSON (default: constraints/merged_barrier_free_constraints.json)')
    parser.add_argument('--episodes', type=int, default=30,
                       help='Number of optimization episodes (default: 30)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--output', default='inference_output',
                       help='Output directory (default: inference_output)')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic actions (less exploration)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip creating visualizations')
    
    args = parser.parse_args()
    
    try:
        # Auto-detect layout file if not specified
        layout_file = args.layout
        if not layout_file:
            print("🔍 Auto-detecting layout file...")
            layout_file = find_layout_file()
            if not layout_file:
                print("❌ No layout file found!")
                print("Please ensure you have a JSON file with room layout in the current directory,")
                print("or specify one with --layout filename.json")
                print("\nLooking for files like:")
                print("  - room_layout.json")
                print("  - room-layout.json") 
                print("  - layout.json")
                print("  - any JSON with 'room' and 'furniture' keys")
                return 1
            else:
                print(f"✓ Found layout file: {layout_file}")
        
        # Load room layout
        if not os.path.exists(layout_file):
            print(f"❌ Layout file not found: {layout_file}")
            return 1
            
        with open(layout_file, 'r') as f:
            room_layout = json.load(f)
        
        # Validate room layout structure
        if 'room' not in room_layout or 'furniture' not in room_layout:
            print("❌ Invalid room layout format. Must contain 'room' and 'furniture' keys.")
            return 1
        inference_engine = BarrierFreeInference(args.model, args.constraints)
        
        # Run optimization
        best_layout = inference_engine.optimize_single_layout(
            room_layout=room_layout,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            use_deterministic=args.deterministic,
            save_visualizations=not args.no_viz,
            output_dir=args.output
        )
        
        print("\n" + "="*60)
        print("🎉 OPTIMIZATION COMPLETE!")
        print("="*60)
        print(f"✓ Best accessibility score: {best_layout['accessibility_score']:.3f}")
        print(f"✓ Layout efficiency: {best_layout['layout_efficiency']:.3f}")
        print(f"✓ Total violations: {best_layout['total_violations']}")
        print(f"✓ Results saved to: {args.output}/")
        print(f"✓ Optimized layout: {args.output}/optimized_layout.json")
        
        if not args.no_viz:
            print(f"✓ Layout visualization: {args.output}/optimized_layout.png")
            print(f"✓ Progress charts: {args.output}/optimization_progress.png")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error during optimization: {e}")
        logger.exception("Detailed error information:")
        return 1


if __name__ == "__main__":
    
    exit(main())