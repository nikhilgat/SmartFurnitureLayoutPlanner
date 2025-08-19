import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
from scipy.ndimage import label as nd_label
import copy  # Added missing import

# Load configuration files
with open("room-layout (1).json", "r") as f:
    config = json.load(f)
with open("barrier_free_constraints_relational.json", "r") as f:
    constraints = json.load(f)

ROOM_WIDTH = config["room"]["width"]
ROOM_HEIGHT = config["room"]["height"]
furniture_items = config["furniture"]
openings = config.get("openings", [])

class InteractiveFurniturePlanner:
    def __init__(self):
        self.clearances = constraints.get("furniture_specific_clearances", {})
        self.pairs = constraints.get("functional_pairs", {})
        self.room_rect = {"x": 0, "y": 0, "width": ROOM_WIDTH, "height": ROOM_HEIGHT}
        self.furniture_groups = {
            "sleeping_zone": ["Bed", "King Bed", "Queen Bed", "Twin Bed", "Bedside Table", "Nightstand"],
            "dining_zone": ["Dining Table", "Kitchen Table", "Dining Chair", "Chair"],
            "living_zone": ["Sofa", "Sectional", "Loveseat", "Coffee Table", "Side Table", "TV Cabinet"],
            "work_zone": ["Desk", "Office Desk", "Writing Desk", "Office Chair", "Desk Chair", "Study Table", "Study Chair"],
            "bathroom_zone": ["Toilet", "Washbasin", "Shower", "Bathtub"],
            "storage_zone": ["Wardrobe", "Closet", "Armoire", "Dresser"]
        }

    def normalize_furniture_name(self, name: str) -> str:
        """Normalize furniture names for consistency."""
        name_map = {
            "bed": "Bed", "sofa": "Sofa", "chair": "Chair", "dining chair": "Dining Chair",
            "table": "Dining Table", "dining table": "Dining Table", "coffee table": "Coffee Table",
            "desk": "Desk", "office desk": "Office Desk", "writing desk": "Writing Desk",
            "office chair": "Office Chair", "desk chair": "Desk Chair",
            "study table": "Study Table", "study chair": "Study Chair",  # Added mappings
            "wardrobe": "Wardrobe", "tv cabinet": "TV Cabinet",
            "sink": "Washbasin", "toilet": "Toilet", "nightstand": "Bedside Table"
        }
        return name_map.get(name.lower(), name.title())

    def get_furniture_dimensions(self, furniture: Dict) -> Tuple[float, float]:
        """Return dimensions considering rotation."""
        if furniture["rotation"] in [90, 270]:
            return furniture["height"], furniture["width"]
        return furniture["width"], furniture["height"]

    def get_opening_rect(self, opening: Dict) -> Dict:
        """Convert wall-based opening to rectangle coordinates."""
        if "x" in opening and "y" in opening:
            return opening
        wall = opening["wall"].lower()
        pos = opening["position"]
        size = opening["size"]
        if wall == "bottom":
            return {"x": pos, "y": 0, "width": size, "height": 1}
        elif wall == "top":
            return {"x": pos, "y": ROOM_HEIGHT - 1, "width": size, "height": 1}
        elif wall == "left":
            return {"x": 0, "y": pos, "width": 1, "height": size}
        elif wall == "right":
            return {"x": ROOM_WIDTH - 1, "y": pos, "width": 1, "height": size}
        raise ValueError(f"Unknown wall: {wall}")

    def check_rectangle_overlap(self, rect1: Dict, rect2: Dict) -> bool:
        """Check if two rectangles overlap."""
        return not (rect1["x"] + rect1["width"] <= rect2["x"] or
                    rect1["x"] >= rect2["x"] + rect2["width"] or
                    rect1["y"] + rect1["height"] <= rect2["y"] or
                    rect1["y"] >= rect2["y"] + rect2["height"])

    def intersect_rect(self, r1: Dict, r2: Dict) -> Dict:
        """Compute intersection rectangle."""
        x = max(r1["x"], r2["x"])
        y = max(r1["y"], r2["y"])
        w = min(r1["x"] + r1["width"], r2["x"] + r2["width"]) - x
        h = min(r1["y"] + r1["height"], r2["y"] + r2["height"]) - y
        if w > 0 and h > 0:
            return {"x": x, "y": y, "width": w, "height": h}
        return None

    def rectangle_overlap_area(self, r1: Dict, r2: Dict) -> float:
        """Compute overlap area between two rectangles."""
        inter = self.intersect_rect(r1, r2)
        return inter["width"] * inter["height"] if inter else 0.0

    def create_clearance_zones(self, furniture: Dict) -> List[Dict]:
        """Generate clearance zones based on furniture type and rotation."""
        w, h = self.get_furniture_dimensions(furniture)
        norm_name = self.normalize_furniture_name(furniture["name"])
        clearance = self.clearances.get(norm_name, {"front_clearance": 90})
        front_clear = clearance.get("front_clearance", 90)
        zones = []
        rotations = {
            0: {"x": furniture["x"], "y": furniture["y"] + h, "width": w, "height": front_clear},
            90: {"x": furniture["x"] + w, "y": furniture["y"], "width": front_clear, "height": h},
            180: {"x": furniture["x"], "y": furniture["y"] - front_clear, "width": w, "height": front_clear},
            270: {"x": furniture["x"] - front_clear, "y": furniture["y"], "width": front_clear, "height": h}
        }
        zone = rotations[furniture["rotation"]]
        zone["type"] = "front"
        zones.append(zone)
        return zones

    def compute_circulation(self, layout: List[Dict]) -> float:
        """Compute circulation penalty (number of disconnected free space components - 1)."""
        res = 10  # grid resolution in cm
        grid_h = int(math.ceil(ROOM_HEIGHT / res))
        grid_w = int(math.ceil(ROOM_WIDTH / res))
        grid = np.ones((grid_h, grid_w))  # 1 = free
        person_rad = 18  # cm for person disk approximation

        for f in layout:
            w, h = self.get_furniture_dimensions(f)
            exp_x = f["x"] - person_rad
            exp_y = f["y"] - person_rad
            exp_w = w + 2 * person_rad
            exp_h = h + 2 * person_rad
            x1 = max(0, int(math.floor(exp_x / res)))
            y1 = max(0, int(math.floor(exp_y / res)))
            x2 = min(grid_w, int(math.ceil((exp_x + exp_w) / res)))
            y2 = min(grid_h, int(math.ceil((exp_y + exp_h) / res)))
            if x2 > x1 and y2 > y1:
                grid[y1:y2, x1:x2] = 0  # occupied

        labeled, num_components = nd_label(grid)
        return max(0, num_components - 1)

    def compute_energy(self, layout: List[Dict]) -> float:
        """Compute energy (cost) based on interior design guidelines."""
        energy = 0.0

        # 1. Penetration and clearance violations
        for i, f in enumerate(layout):
            w, h = self.get_furniture_dimensions(f)
            f_rect = {"x": f["x"], "y": f["y"], "width": w, "height": h}
            # Room boundary penetration
            inter = self.intersect_rect(f_rect, self.room_rect)
            inter_area = inter["width"] * inter["height"] if inter else 0.0
            penetration = (w * h) - inter_area
            energy += 1000 * penetration  # High penalty for out-of-bounds

            # Overlaps with other furniture
            for j in range(i + 1, len(layout)):
                g = layout[j]
                g_w, g_h = self.get_furniture_dimensions(g)
                g_rect = {"x": g["x"], "y": g["y"], "width": g_w, "height": g_h}
                overlap = self.rectangle_overlap_area(f_rect, g_rect)
                energy += 1000 * overlap  # High penalty for overlaps

            # Clearance violations
            zones = self.create_clearance_zones(f)
            for zone in zones:
                zone_area = zone["width"] * zone["height"]
                inter = self.intersect_rect(zone, self.room_rect)
                inter_area = inter["width"] * inter["height"] if inter else 0.0
                energy += 500 * (zone_area - inter_area)  # Penalty for clearance out-of-bounds
                for j, g in enumerate(layout):
                    if j != i:
                        g_w, g_h = self.get_furniture_dimensions(g)
                        g_rect = {"x": g["x"], "y": g["y"], "width": g_w, "height": g_h}
                        overlap = self.rectangle_overlap_area(zone, g_rect)
                        energy += 500 * overlap  # Penalty for clearance overlap

            # Opening blockage
            for opening in openings:
                opening_rect = self.get_opening_rect(opening)
                if self.check_rectangle_overlap(f_rect, opening_rect):
                    energy += 1000  # High penalty for blocking openings

        # 2. Pairwise relationship distances
        for i, f in enumerate(layout):
            norm = self.normalize_furniture_name(f["name"])
            if norm in self.pairs:
                cons = self.pairs[norm]
                for partner_type in cons.get("required_partners", []) + cons.get("optional_partners", []):
                    partners = [g for g in layout[i+1:] if self.normalize_furniture_name(g["name"]) == partner_type]
                    if partners:
                        closest = min(partners, key=lambda p: self.calculate_distance(f, p))
                        d = self.calculate_distance(f, closest )
                        min_d = cons.get("min_distance", 30)
                        max_d = cons.get("max_distance", 200)
                        if d < min_d:
                            energy += 100 * (min_d - d) ** 2
                        elif d > max_d:
                            energy += 100 * (d - max_d) ** 2

        # 3. Visual balance (variance of weighted positions)
        total_area = 0.0
        mean_x, mean_y = 0.0, 0.0
        for f in layout:
            w, h = self.get_furniture_dimensions(f)
            a = w * h
            cx = f["x"] + w / 2
            cy = f["y"] + h / 2
            mean_x += a * cx
            mean_y += a * cy
            total_area += a
        if total_area > 0:
            mean_x /= total_area
            mean_y /= total_area
        var = 0.0
        for f in layout:
            w, h = self.get_furniture_dimensions(f)
            a = w * h
            cx = f["x"] + w / 2
            cy = f["y"] + h / 2
            var += a * ((cx - mean_x) ** 2 + (cy - mean_y) ** 2)
        if total_area > 0:
            var /= total_area
            energy += var / 1000  # Scaled to balance with other terms

        # 4. Circulation penalty
        energy += 1000 * self.compute_circulation(layout)

        # 5. Alignment with walls
        for f in layout:
            w, h = self.get_furniture_dimensions(f)
            if f["x"] <= 10 or f["x"] + w >= ROOM_WIDTH - 10 or f["y"] <= 10 or f["y"] + h >= ROOM_HEIGHT - 10:
                energy -= 50  # Bonus for wall alignment
            if f["rotation"] % 90 == 0:
                energy -= 20  # Bonus for orthogonal alignment

        return energy

    def calculate_distance(self, f1: Dict, f2: Dict) -> float:
        """Calculate center-to-center distance between furniture items."""
        w1, h1 = self.get_furniture_dimensions(f1)
        w2, h2 = self.get_furniture_dimensions(f2)
        cx1 = f1["x"] + w1 / 2
        cy1 = f1["y"] + h1 / 2
        cx2 = f2["x"] + w2 / 2
        cy2 = f2["y"] + h2 / 2
        return math.hypot(cx1 - cx2, cy1 - cy2)

    def optimize_layout(self, initial_layout: List[Dict]) -> List[Dict]:
        """Optimize layout using MCMC sampling."""
        layout = copy.deepcopy(initial_layout)
        best_layout = copy.deepcopy(layout)
        best_energy = self.compute_energy(layout)
        T = 100.0  # Initial temperature
        steps = 2000  # Number of MCMC iterations

        for _ in range(steps):
            idx = random.randint(0, len(layout) - 1)
            f = layout[idx]
            old_x, old_y, old_rot = f["x"], f["y"], f["rotation"]

            # Propose changes
            f["x"] += random.gauss(0, 20)
            f["y"] += random.gauss(0, 20)
            if random.random() < 0.1:
                f["rotation"] = random.choice([0, 90, 180, 270])

            # Clip to room boundaries
            w, h = self.get_furniture_dimensions(f)
            f["x"] = max(0, min(f["x"], ROOM_WIDTH - w))
            f["y"] = max(0, min(f["y"], ROOM_HEIGHT - h))

            new_energy = self.compute_energy(layout)
            delta = new_energy - best_energy

            if delta < 0 or random.random() < math.exp(-delta / T):
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_layout = copy.deepcopy(layout)
            else:
                f["x"], f["y"], f["rotation"] = old_x, old_y, old_rot

            T *= 0.995  # Cool down

        return best_layout

def load_initial_layout() -> List[Dict]:
    """Load initial furniture layout from JSON."""
    return [
        {
            "name": item["name"],
            "x": item["x"],
            "y": item["y"],
            "rotation": item.get("rotation", 0),
            "width": item["width"],
            "height": item["height"]
        } for item in furniture_items
    ]

def visualize_layout(before_layout: List[Dict], after_layout: List[Dict], planner: InteractiveFurniturePlanner):
    """Visualize initial and optimized layouts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
    layouts = [(before_layout, ax1, "Initial Layout"), (after_layout, ax2, "Optimized Layout")]

    for layout, ax, title in layouts:
        ax.set_xlim(0, ROOM_WIDTH)
        ax.set_ylim(0, ROOM_HEIGHT)
        ax.set_title(f"{title} (Energy: {planner.compute_energy(layout):.1f})")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Draw clearance zones
        for f in layout:
            for zone in planner.create_clearance_zones(f):
                color = "orange" if zone["type"] == "front" else "yellow"
                alpha = 0.2 if zone["type"] == "front" else 0.1
                rect = patches.Rectangle(
                    (zone["x"], zone["y"]), zone["width"], zone["height"],
                    linewidth=1, edgecolor=color, facecolor=color, alpha=alpha, linestyle="--"
                )
                ax.add_patch(rect)

        # Draw furniture
        for f in layout:
            w, h = planner.get_furniture_dimensions(f)
            group = next((g for g, items in planner.furniture_groups.items() if planner.normalize_furniture_name(f["name"]) in items), "miscellaneous")
            group_colors = {
                "sleeping_zone": "lightblue", "dining_zone": "lightcoral", "living_zone": "lightgreen",
                "work_zone": "lightyellow", "bathroom_zone": "lightcyan", "storage_zone": "plum",
                "miscellaneous": "lightgray"
            }
            rect = patches.Rectangle(
                (f["x"], f["y"]), w, h, linewidth=2, edgecolor="black",
                facecolor=group_colors.get(group, "lightgray"), alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(f["x"] + w/2, f["y"] + h/2, f["name"], ha="center", va="center", fontsize=8, weight="bold")

        # Draw openings
        for opening in openings:
            opening_rect = planner.get_opening_rect(opening)
            rect = patches.Rectangle(
                (opening_rect["x"], opening_rect["y"]), opening_rect["width"], opening_rect["height"],
                linewidth=3, edgecolor="green", facecolor="lightgreen", alpha=0.6
            )
            ax.add_patch(rect)
            ax.text(
                opening_rect["x"] + opening_rect["width"]/2, opening_rect["y"] + opening_rect["height"]/2,
                "DOOR", ha="center", va="center", fontweight="bold", color="darkgreen"
            )

        ax.set_xlabel("Width (cm)")
        ax.set_ylabel("Height (cm)")

    plt.tight_layout()
    plt.show()

def save_layout(layout: List[Dict], filename: str = "mcmc_optimized_layout.json"):
    """Save the optimized layout to JSON."""
    output_data = {
        "method": "MCMC-Based Furniture Layout",
        "room": {"width": ROOM_WIDTH, "height": ROOM_HEIGHT},
        "furniture": layout,
        "openings": openings,
        "features": [
            "Interior design guideline optimization",
            "Accessibility compliance",
            "Stochastic sampling for layout exploration",
            "Functional and visual balance"
        ]
    }
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Optimized layout saved to {filename}")

if __name__ == "__main__":
    print("🏠 Optimizing Furniture Layout with MCMC...")
    print(f"Room dimensions: {ROOM_WIDTH}cm x {ROOM_HEIGHT}cm")
    print(f"Number of furniture items: {len(furniture_items)}")

    planner = InteractiveFurniturePlanner()
    initial_layout = load_initial_layout()
    print("\n🎯 Generating optimized layout...")
    optimized_layout = planner.optimize_layout(initial_layout)

    print("\n🎨 Visualizing layouts...")
    visualize_layout(initial_layout, optimized_layout, planner)
    save_layout(optimized_layout)

    print("\n✅ Optimization complete!")
    print("🔗 Features:")
    print("   • Functional clearance and accessibility")
    print("   • Pairwise relationship optimization")
    print("   • Visual balance and wall alignment")
    print("   • Circulation path connectivity")