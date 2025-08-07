import os
import re
import json
import torch
import multiprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from furniture_env import EnhancedFurnitureEnv  


class LayoutExtractionCallback(BaseCallback):
    """Custom callback to extract and save best layouts during training"""

    def __init__(self, save_freq: int = 10000, save_path: str = "./layouts/", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self._save_best_layouts()
        return True

    def _save_best_layouts(self):
        try:
            best_layouts = [get_best_layout() for get_best_layout in self.training_env.get_attr('get_best_layout')]
            for i, layout in enumerate(best_layouts):
                if layout:
                    filename = os.path.join(self.save_path, f"best_layout_env{i}_step{self.num_timesteps}.json")
                    with open(filename, 'w') as f:
                        json.dump(layout, f, indent=2)
            if self.verbose:
                print(f"💾 Best layouts saved at step {self.num_timesteps}")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Could not save layouts: {e}")


class CustomCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(save_freq, save_path, name_prefix, verbose)

    def _on_step(self) -> bool:
        result = super()._on_step()
        # Save VecNormalize stats alongside model checkpoint
        if self.n_calls % self.save_freq == 0:
            vecnorm_path = os.path.join(self.save_path, f"{self.name_prefix}_vecnormalize_{self.num_timesteps}_steps.pkl")
            # Unwrap VecNormalize from SubprocVecEnv
            env = self.training_env
            while hasattr(env, "venv"):
                env = env.venv
            if isinstance(env, VecNormalize):
                env.save(vecnorm_path)
                if self.verbose:
                    print(f"💾 Saved VecNormalize stats to {vecnorm_path}")
        return result


def make_env(furniture_list=None, room_width=800, room_height=600, use_discrete=True):
    def _init():
        return EnhancedFurnitureEnv(
            room_width=room_width,
            room_height=room_height,
            furniture_list=furniture_list,
            max_steps=1000,
            use_discrete_actions=use_discrete
        )
    return _init


def extract_steps(filename):
    match = re.search(r"_(\d+)_steps\.zip", filename)
    return int(match.group(1)) if match else -1


def train_optimal_layout_model(
    furniture_list=None,
    room_width=800,
    room_height=600,
    total_timesteps=2_000_000,
    n_envs=8,
    use_discrete_actions=True
):
    print("🚀 Starting Enhanced Furniture Layout Training")
    print(f"Room: {room_width}x{room_height}, Furniture: {len(furniture_list) if furniture_list else 5}, Action Space: {'Discrete' if use_discrete_actions else 'Continuous'}")

    model_path = "optimal_furniture_model.zip"  # fallback path (not used for resuming)
    vecnorm_path = "optimal_vecnormalize.pkl"  # fallback path
    layouts_dir = "./optimal_layouts/"
    checkpoint_dir = "./optimal_checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = SubprocVecEnv([make_env(furniture_list, room_width, room_height, use_discrete_actions) for _ in range(n_envs)])

    # Find latest checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith("_steps.zip")]
    latest_model_path = None
    latest_vecnorm_path = None
    resume_timesteps = 0

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=extract_steps)
        resume_timesteps = extract_steps(latest_checkpoint)
        latest_model_path = os.path.join(checkpoint_dir, latest_checkpoint)
        latest_vecnorm_path = os.path.join(checkpoint_dir, f"optimal_furniture_vecnormalize_{resume_timesteps}_steps.pkl")
        print(f"🔄 Resuming from checkpoint: {latest_checkpoint} @ {resume_timesteps} steps")
    else:
        print("ℹ️ No checkpoints found. Starting fresh.")

    # Load VecNormalize or create new
    if latest_vecnorm_path and os.path.exists(latest_vecnorm_path):
        print(f"✅ Loaded VecNormalize from {latest_vecnorm_path}")
        env = VecNormalize.load(latest_vecnorm_path, env)
        env.training = True
        env.norm_reward = True
    else:
        print("🆕 Creating new VecNormalize")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

    env = VecMonitor(env)

    policy_kwargs = dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]), activation_fn=torch.nn.ReLU)

    # Load or create PPO model
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"✅ Loaded PPO model from {latest_model_path}")
        model = PPO.load(latest_model_path, env=env)
        print(f"⏱ Resuming training from timestep: {model.num_timesteps}")
    else:
        print("🆕 Initializing new PPO model")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./optimal_furniture_tb/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            policy_kwargs=policy_kwargs,
            device="auto"
        )

    layout_callback = LayoutExtractionCallback(save_freq=50000, save_path=layouts_dir, verbose=1)
    checkpoint_callback = CustomCheckpointCallback(save_freq=100000, save_path=checkpoint_dir, name_prefix="optimal_furniture", verbose=1)

    eval_env = SubprocVecEnv([make_env(furniture_list, room_width, room_height, use_discrete_actions) for _ in range(min(4, n_envs))])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./optimal_best_model/",
        log_path="./optimal_logs/",
        eval_freq=max(50000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )

    # Calculate remaining timesteps to train
    remaining_timesteps = total_timesteps - model.num_timesteps
    if remaining_timesteps <= 0:
        print("⚠️ Model already trained for equal or more timesteps than total_timesteps. Exiting.")
        return model, env

    print(f"🎯 Training for remaining {remaining_timesteps:,} steps...")
    model.learn(total_timesteps=remaining_timesteps, callback=[layout_callback, checkpoint_callback, eval_callback], reset_num_timesteps=False)

    # Save final model and VecNormalize stats
    final_model_path = os.path.join(checkpoint_dir, "optimal_furniture_final.zip")
    final_vecnorm_path = os.path.join(checkpoint_dir, "optimal_furniture_vecnormalize_final.pkl")
    model.save(final_model_path)
    env.save(final_vecnorm_path)

    print("✅ Final model and VecNormalize stats saved!")
    return model, env


def extract_best_layout(model, env, num_attempts=50):
    print(f"🔍 Extracting best layout from {num_attempts} episodes")
    best_layout = None
    best_reward = float('-inf')
    perfect_layouts = []

    for attempt in range(num_attempts):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]

            if info[0].get('perfect_layout', False):
                layout = env.get_attr('get_layout_json')[0]()
                perfect_layouts.append(layout)
                print(f"✨ Found perfect layout at attempt {attempt + 1}")

        final_layout = env.get_attr('get_layout_json')[0]()
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_layout = final_layout

        if attempt % 10 == 0:
            print(f"Attempt {attempt + 1}/{num_attempts} | Best reward so far: {best_reward:.2f}")

    if perfect_layouts:
        with open("final_perfect_layout.json", 'w') as f:
            json.dump(perfect_layouts[0], f, indent=2)
        return perfect_layouts[0]
    else:
        with open("final_best_layout.json", 'w') as f:
            json.dump(best_layout, f, indent=2)
        return best_layout


if __name__ == "__main__":
    multiprocessing.freeze_support()

    with open("barrierfreiRL/barrier_free_constraints.json", 'r') as f:
        constraints = json.load(f)

    custom_furniture = [
        {"name": "Table", "width": 120, "height": 70, "zHeight": 75},
        {"name": "Sofa", "width": 200, "height": 90, "zHeight": 85},
        {"name": "Chair", "width": 50, "height": 50, "zHeight": 90},
        {"name": "Bed", "width": 180, "height": 200, "zHeight": 55},
        {"name": "Wardrobe", "width": 150, "height": 60, "zHeight": 200}
    ]

    model, env = train_optimal_layout_model(
        furniture_list=custom_furniture,
        room_width=800,
        room_height=600,
        total_timesteps=3_000_000,
        n_envs=8,
        use_discrete_actions=True
    )

    final_layout = extract_best_layout(model, env, num_attempts=100)

    print("\n🏁 Training complete!")
    print("📄 Final layout saved to 'final_perfect_layout.json' or 'final_best_layout.json'")
    print("\n📊 Layout Summary:")
    print(json.dumps(final_layout, indent=2))
