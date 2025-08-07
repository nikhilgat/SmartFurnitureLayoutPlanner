import random
import math
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load config JSON
with open("room-layout.json", "r") as f:
    config = json.load(f)

ROOM_WIDTH = config["room"]["width"]
ROOM_HEIGHT = config["room"]["height"]
furniture_items = config["furniture"]
openings = config.get("openings", [])
CLEARANCE = 90  # cm clearance for accessibility

# Furniture names that require clearance zones
def needs_clearance(name):
    return name.lower() in ["bed", "sofa", "toilet", "sink", "wardrobe"]

# Load the original input layout exactly as initial individual
def load_initial_layout():
    layout = []
    for item in furniture_items:
        layout.append({
            "name": item["name"],
            "x": item["x"],
            "y": item["y"],
            "rotation": item.get("rotation", 0),
            "width": item["width"],
            "height": item["height"]
        })
    return layout

# Create population as copies of the original layout (no randomizing positions)
def create_population(base_layout, population_size):
    return [[dict(f) for f in base_layout] for _ in range(population_size)]

# Rectangle overlap check
def overlap(a, b):
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)

# Fitness function evaluating collisions, boundaries, clearance, and blocking openings
def evaluate(layout):
    score = 1000

    # Pairwise checks for overlaps including clearance zones
    for i in range(len(layout)):
        for j in range(i + 1, len(layout)):
            a = layout[i].copy()
            b = layout[j].copy()

            a_w, a_h = (a["width"], a["height"]) if a["rotation"] == 0 else (a["height"], a["width"])
            b_w, b_h = (b["width"], b["height"]) if b["rotation"] == 0 else (b["height"], b["width"])

            if needs_clearance(a["name"]):
                a["x"] -= CLEARANCE / 2
                a["y"] -= CLEARANCE / 2
                a_w += CLEARANCE
                a_h += CLEARANCE

            if needs_clearance(b["name"]):
                b["x"] -= CLEARANCE / 2
                b["y"] -= CLEARANCE / 2
                b_w += CLEARANCE
                b_h += CLEARANCE

            if overlap({"x": a["x"], "y": a["y"], "width": a_w, "height": a_h},
                       {"x": b["x"], "y": b["y"], "width": b_w, "height": b_h}):
                score -= 300

    # Check if furniture is inside room bounds
    for obj in layout:
        w, h = (obj["width"], obj["height"]) if obj["rotation"] == 0 else (obj["height"], obj["width"])
        if not (0 <= obj["x"] <= ROOM_WIDTH - w and 0 <= obj["y"] <= ROOM_HEIGHT - h):
            score -= 100

    # Check if furniture blocks openings
    for obj in layout:
        w, h = (obj["width"], obj["height"]) if obj["rotation"] == 0 else (obj["height"], obj["width"])
        obj_rect = {"x": obj["x"], "y": obj["y"], "width": w, "height": h}
        for opening in openings:
            if overlap(obj_rect, opening):
                score -= 300

    # Small reward for spacing out furniture
    for i in range(len(layout)):
        for j in range(i + 1, len(layout)):
            xi, yi = layout[i]["x"], layout[i]["y"]
            xj, yj = layout[j]["x"], layout[j]["y"]
            dist = math.hypot(xi - xj, yi - yj)
            score += dist * 0.01

    return score

# Mutation: move or rotate one furniture item within bounds
def mutate(layout):
    idx = random.randint(0, len(layout) - 1)
    item = dict(layout[idx])  # copy to mutate

    w, h = (item["width"], item["height"]) if item["rotation"] == 0 else (item["height"], item["width"])
    item["x"] = random.uniform(0, ROOM_WIDTH - w)
    item["y"] = random.uniform(0, ROOM_HEIGHT - h)
    item["rotation"] = random.choice([0, 90])

    new_layout = [dict(f) for f in layout]
    new_layout[idx] = item
    return new_layout

# Crossover: child takes furniture positions randomly from parents
def crossover(p1, p2):
    return [random.choice([p1[i], p2[i]]) for i in range(len(p1))]

# Genetic algorithm main loop
def genetic_algorithm(initial_layout, generations=50, population_size=20):
    population = create_population(initial_layout, population_size)

    for gen in range(generations):
        population.sort(key=evaluate, reverse=True)
        print(f"Generation {gen}: Best fitness = {evaluate(population[0]):.2f}")

        next_gen = population[:5]  # Elitism: carry top 5 directly

        while len(next_gen) < population_size:
            p1, p2 = random.choices(population[:10], k=2)
            child = crossover(p1, p2)
            if random.random() < 0.3:
                child = mutate(child)
            next_gen.append(child)

        population = next_gen

    return population[0]

# Visualization of before & after with clearance and openings
def visualize_comparison(before_layout, after_layout):
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    titles = ["Before Optimization", "After Optimization"]
    layouts = [before_layout, after_layout]

    for i in range(2):
        ax = axs[i]
        ax.set_xlim(0, ROOM_WIDTH)
        ax.set_ylim(0, ROOM_HEIGHT)
        ax.set_title(f"{titles[i]} (Fitness: {evaluate(layouts[i]):.1f})")
        ax.set_aspect('equal')
        ax.grid(True)

        for obj in layouts[i]:
            w, h = (obj["width"], obj["height"]) if obj["rotation"] == 0 else (obj["height"], obj["width"])
            rect = patches.Rectangle((obj["x"], obj["y"]), w, h,
                                     linewidth=2, edgecolor='black',
                                     facecolor='lightblue', alpha=0.7)
            ax.add_patch(rect)
            ax.text(obj["x"] + w / 2, obj["y"] + h / 2,
                    obj["name"], ha='center', va='center',
                    fontsize=10, weight='bold')

            # Draw clearance zone if needed
            if needs_clearance(obj["name"]):
                clr = patches.Rectangle(
                    (obj["x"] - CLEARANCE/2, obj["y"] - CLEARANCE/2),
                    w + CLEARANCE, h + CLEARANCE,
                    linewidth=1, edgecolor='orange',
                    facecolor='none', linestyle='--')
                ax.add_patch(clr)

        # Draw openings (doors/windows)
        for op in openings:
            door = patches.Rectangle((op["x"], op["y"]), op["width"], op["height"],
                                     linewidth=2, edgecolor='green',
                                     facecolor='lightgreen', alpha=0.4)
            ax.add_patch(door)

        ax.set_xlabel("Width (cm)")
        ax.set_ylabel("Height (cm)")

    plt.tight_layout()
    plt.show()

# ===== RUN =====

initial_layout = load_initial_layout()
best_layout = genetic_algorithm(initial_layout, 100, 200)

# Save optimized layout to JSON file
with open("optimized_layout.json", "w") as f:
    json.dump(best_layout, f, indent=2)

print("Optimized layout saved to optimized_layout.json")

visualize_comparison(initial_layout, best_layout)
