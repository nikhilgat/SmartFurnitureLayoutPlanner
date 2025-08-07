import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from furniture_env import FurnitureEnv
import torch
import numpy as np
import os

# === CONFIG ===
MODEL_PATH = "ppo_furniture_model.zip"
VECNORM_PATH = "vecnormalize.pkl"
INPUT_JSON = "room-layout.json"           # your input layout file
OUTPUT_JSON = "optimized_layout.json"      # will be saved here

def load_input_layout(path):
    with open(path) as f:
        return json.load(f)

def draw_layout(ax, furniture, room, title):
    ax.set_title(title)
    ax.set_xlim(0, room["width"])
    ax.set_ylim(0, room["height"])
    ax.set_aspect("equal")

    for item in furniture:
        rect = patches.Rectangle(
            (item["x"], item["y"]),
            item["width"],
            item["height"],
            angle=item.get("rotation", 0),
            edgecolor='blue',
            facecolor='cyan',
            alpha=0.5
        )
        ax.add_patch(rect)
        ax.text(
            item["x"] + item["width"]/2,
            item["y"] + item["height"]/2,
            item["name"],
            ha='center',
            va='center',
            fontsize=8
        )

def run_inference(input_data):
    room = input_data["room"]
    furniture_list = input_data["furniture"]

    # === Set up environment ===
    env = FurnitureEnv(
        room_width=room["width"],
        room_height=room["height"],
        num_items=len(furniture_list),
        max_steps=100,
        auto_curriculum=False
    )
    env.furniture = furniture_list.copy()  # Set fixed furniture layout

    vec_env = DummyVecEnv([lambda: env])

    # === Load trained PPO model ===
    print("✅ Loading model...")
    model = PPO.load(MODEL_PATH, env=vec_env)

    obs = vec_env.reset()

    for _ in range(100):  # run inference steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = vec_env.step(action)
        if done[0]:
            break

    optimized = env.furniture

    return room, furniture_list, optimized

def save_output_layout(room, furniture, path):
    output = {
        "room": room,
        "furniture": furniture,
        "openings": []
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Optimized layout saved to: {path}")

if __name__ == "__main__":
    input_data = load_input_layout(INPUT_JSON)
    room, original_furniture, optimized_furniture = run_inference(input_data)

    # === Save result
    save_output_layout(room, optimized_furniture, OUTPUT_JSON)

    # === Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    draw_layout(ax1, original_furniture, room, "Original Layout")
    draw_layout(ax2, optimized_furniture, room, "Optimized Layout")
    plt.tight_layout()
    plt.show()
