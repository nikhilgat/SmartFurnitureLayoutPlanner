import random
import math
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load config JSON
with open("room-layout.json", "r") as f:
    config = json.load(f)

# Load barrier-free constraints
with open("merged_barrier_free_constraints.json", "r") as f:
    bf_constraints = json.load(f)

ROOM_WIDTH = config["room"]["width"]
ROOM_HEIGHT = config["room"]["height"]
furniture_items = config["furniture"]
openings = config.get("openings", [])

class BarrierFreePlanner:
    def __init__(self):
        self.clearance_reqs = bf_constraints["clearance_requirements"]
        self.furniture_clearances = bf_constraints["furniture_specific_clearances"]
        self.relationships = bf_constraints["furniture_relationships"]
        self.ergonomics = bf_constraints["ergonomic_constraints"]
        self.accessibility = bf_constraints["accessibility_enhancements"]
    
    def normalize_furniture_name(self, name):
        """Normalize furniture names for constraint lookup"""
        name_map = {
            "bed": "Bed", "sofa": "Sofa", "chair": "Chair", "dining chair": "Dining Chair",
            "table": "Dining Table", "dining table": "Dining Table", "coffee table": "Coffee Table",
            "desk": "Desk", "wardrobe": "Wardrobe", "tv cabinet": "TV Cabinet",
            "sink": "Washbasin", "toilet": "Toilet", "armchair": "Armchair"
        }
        return name_map.get(name.lower(), name.title())
    
    def get_furniture_dimensions(self, furniture):
        """Get actual dimensions considering rotation"""
        if furniture["rotation"] in [90, 270]:
            return furniture["height"], furniture["width"]
        return furniture["width"], furniture["height"]
    
    def create_clearance_zones(self, furniture):
        """Create accessibility clearance zones around furniture"""
        w, h = self.get_furniture_dimensions(furniture)
        norm_name = self.normalize_furniture_name(furniture["name"])
        
        if norm_name not in self.furniture_clearances:
            # Default clearance
            return [{
                "x": furniture["x"] - 45, "y": furniture["y"] - 45,
                "width": w + 90, "height": h + 90, "type": "general"
            }]
        
        clearance = self.furniture_clearances[norm_name]
        zones = []
        
        # Front clearance (most important)
        front_clear = clearance.get("front_clearance", 90)
        
        if furniture["rotation"] == 0:  # Facing up
            zones.append({
                "x": furniture["x"], "y": furniture["y"] + h,
                "width": w, "height": front_clear, "type": "front"
            })
        elif furniture["rotation"] == 90:  # Facing right
            zones.append({
                "x": furniture["x"] + w, "y": furniture["y"],
                "width": front_clear, "height": h, "type": "front"
            })
        elif furniture["rotation"] == 180:  # Facing down
            zones.append({
                "x": furniture["x"], "y": furniture["y"] - front_clear,
                "width": w, "height": front_clear, "type": "front"
            })
        else:  # Facing left
            zones.append({
                "x": furniture["x"] - front_clear, "y": furniture["y"],
                "width": front_clear, "height": h, "type": "front"
            })
        
        return zones
    
    def check_rectangle_overlap(self, rect1, rect2):
        """Check if two rectangles overlap"""
        return not (rect1["x"] + rect1["width"] <= rect2["x"] or
                   rect1["x"] >= rect2["x"] + rect2["width"] or
                   rect1["y"] + rect1["height"] <= rect2["y"] or
                   rect1["y"] >= rect2["y"] + rect2["height"])
    
    def calculate_distance(self, furniture1, furniture2):
        """Calculate distance between centers of two furniture items"""
        w1, h1 = self.get_furniture_dimensions(furniture1)
        w2, h2 = self.get_furniture_dimensions(furniture2)
        
        center1_x = furniture1["x"] + w1 / 2
        center1_y = furniture1["y"] + h1 / 2
        center2_x = furniture2["x"] + w2 / 2
        center2_y = furniture2["y"] + h2 / 2
        
        return math.hypot(center2_x - center1_x, center2_y - center1_y)
    
    def is_position_valid(self, furniture, layout):
        """Check if furniture position is valid"""
        w, h = self.get_furniture_dimensions(furniture)
        
        # Check room boundaries
        if not (0 <= furniture['x'] <= ROOM_WIDTH - w and 0 <= furniture['y'] <= ROOM_HEIGHT - h):
            return False
        
        # Check overlaps with existing furniture (including clearances)
        furniture_rect = {'x': furniture['x'], 'y': furniture['y'], 'width': w, 'height': h}
        
        for existing in layout:
            if existing == furniture:
                continue
                
            existing_w, existing_h = self.get_furniture_dimensions(existing)
            existing_rect = {'x': existing['x'], 'y': existing['y'], 'width': existing_w, 'height': existing_h}
            
            if self.check_rectangle_overlap(furniture_rect, existing_rect):
                return False
            
            # Check clearance overlaps
            clearance_zones = self.create_clearance_zones(existing)
            for zone in clearance_zones:
                if self.check_rectangle_overlap(furniture_rect, zone):
                    return False
        
        # Check opening blockage
        for opening in openings:
            if self.check_rectangle_overlap(furniture_rect, opening):
                return False
        
        return True
    
    def has_wheelchair_access(self, furniture, layout):
        """Check if furniture has adequate wheelchair access"""
        clearance_zones = self.create_clearance_zones(furniture)
        
        for zone in clearance_zones:
            if zone["type"] == "front":
                # Check if zone is clear
                for other in layout:
                    if other == furniture:
                        continue
                    
                    w, h = self.get_furniture_dimensions(other)
                    other_rect = {"x": other["x"], "y": other["y"], "width": w, "height": h}
                    
                    if self.check_rectangle_overlap(zone, other_rect):
                        return False
        
        return True
    
    def evaluate_layout(self, layout):
        """Comprehensive layout evaluation"""
        score = 1000
        
        # 1. Accessibility compliance
        accessible_count = sum(1 for f in layout if self.has_wheelchair_access(f, layout))
        accessibility_score = (accessible_count / len(layout)) * 300
        score += accessibility_score
        
        # 2. Functional relationships
        relationship_score = self.evaluate_relationships(layout)
        score += relationship_score
        
        # 3. Space efficiency
        space_efficiency = self.calculate_space_efficiency(layout)
        score += space_efficiency
        
        # 4. Wall proximity bonus (furniture against walls)
        wall_bonus = sum(self.calculate_wall_proximity(f) for f in layout)
        score += wall_bonus
        
        # 5. Circulation space
        circulation_score = self.evaluate_circulation(layout)
        score += circulation_score
        
        return score
    
    def evaluate_relationships(self, layout):
        """Evaluate functional relationships between furniture"""
        score = 0
        functional_pairs = self.relationships["functional_pairs"]
        
        for primary_name, constraints in functional_pairs.items():
            primary_items = [f for f in layout if self.normalize_furniture_name(f["name"]) == primary_name]
            
            for primary_item in primary_items:
                for partner_type in constraints.get("required_partners", []):
                    partner_items = [f for f in layout if self.normalize_furniture_name(f["name"]) == partner_type]
                    
                    if partner_items:
                        # Find closest partner
                        closest_partner = min(partner_items, key=lambda p: self.calculate_distance(primary_item, p))
                        distance = self.calculate_distance(primary_item, closest_partner)
                        
                        min_dist = constraints.get("min_distance", 30)
                        max_dist = constraints.get("max_distance", 200)
                        
                        if min_dist <= distance <= max_dist:
                            score += 100
                        else:
                            score -= 50
        
        return score
    
    def calculate_space_efficiency(self, layout):
        """Calculate how efficiently the space is used"""
        total_furniture_area = sum(
            self.get_furniture_dimensions(f)[0] * self.get_furniture_dimensions(f)[1] 
            for f in layout
        )
        room_area = ROOM_WIDTH * ROOM_HEIGHT
        utilization_ratio = total_furniture_area / room_area
        
        # Optimal utilization is around 30-40% for accessible design
        if 0.25 <= utilization_ratio <= 0.4:
            return 100
        elif utilization_ratio < 0.25:
            return 50 - (0.25 - utilization_ratio) * 200
        else:
            return 50 - (utilization_ratio - 0.4) * 100
    
    def calculate_wall_proximity(self, furniture):
        """Calculate bonus for furniture placed against walls"""
        bonus = 0
        w, h = self.get_furniture_dimensions(furniture)
        
        # Bonus for furniture against walls (especially large items)
        norm_name = self.normalize_furniture_name(furniture["name"])
        if norm_name in ['Bed', 'Wardrobe', 'Sofa']:
            if furniture['x'] <= 10:  # Against left wall
                bonus += 20
            if furniture['x'] + w >= ROOM_WIDTH - 10:  # Against right wall
                bonus += 20
            if furniture['y'] <= 10:  # Against bottom wall
                bonus += 15
            if furniture['y'] + h >= ROOM_HEIGHT - 10:  # Against top wall
                bonus += 15
        
        return bonus
    
    def evaluate_circulation(self, layout):
        """Evaluate circulation space quality"""
        # Check for main circulation path through center
        center_rect = {
            'x': ROOM_WIDTH * 0.3,
            'y': ROOM_HEIGHT * 0.3,
            'width': ROOM_WIDTH * 0.4,
            'height': ROOM_HEIGHT * 0.4
        }
        
        # Bonus if center area is relatively clear
        blocking_furniture = 0
        for furniture in layout:
            w, h = self.get_furniture_dimensions(furniture)
            furniture_rect = {'x': furniture['x'], 'y': furniture['y'], 'width': w, 'height': h}
            
            if self.check_rectangle_overlap(center_rect, furniture_rect):
                blocking_furniture += 1
        
        # Less blocking = better circulation
        return max(0, 100 - blocking_furniture * 20)
    
    def mutate_layout(self, layout, mutation_rate=0.3):
        """Mutate layout by moving or rotating furniture"""
        new_layout = [dict(f) for f in layout]
        
        for furniture in new_layout:
            if random.random() < mutation_rate:
                mutation_type = random.choice(["move", "rotate", "fine_tune"])
                
                if mutation_type == "move":
                    self.try_random_position(furniture, new_layout)
                elif mutation_type == "rotate":
                    furniture["rotation"] = random.choice([0, 90, 180, 270])
                elif mutation_type == "fine_tune":
                    self.fine_tune_position(furniture, new_layout)
        
        return new_layout
    
    def try_random_position(self, furniture, layout):
        """Try to place furniture at a random valid position"""
        w, h = self.get_furniture_dimensions(furniture)
        
        for _ in range(20):  # 20 attempts
            new_x = random.uniform(0, ROOM_WIDTH - w)
            new_y = random.uniform(0, ROOM_HEIGHT - h)
            
            test_furniture = {**furniture, 'x': new_x, 'y': new_y}
            
            if self.is_position_valid(test_furniture, layout):
                furniture['x'] = new_x
                furniture['y'] = new_y
                break
    
    def fine_tune_position(self, furniture, layout):
        """Make small adjustments to furniture position"""
        w, h = self.get_furniture_dimensions(furniture)
        
        # Small random adjustment
        dx = random.uniform(-20, 20)
        dy = random.uniform(-20, 20)
        
        new_x = max(0, min(ROOM_WIDTH - w, furniture['x'] + dx))
        new_y = max(0, min(ROOM_HEIGHT - h, furniture['y'] + dy))
        
        test_furniture = {**furniture, 'x': new_x, 'y': new_y}
        
        if self.is_position_valid(test_furniture, layout):
            furniture['x'] = new_x
            furniture['y'] = new_y
    
    def crossover(self, parent1, parent2):
        """Create child layout by combining two parent layouts"""
        child = []
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(dict(parent1[i]))
            else:
                child.append(dict(parent2[i]))
        
        return child

def create_initial_population(base_layout, population_size, planner):
    """Create initial population with random valid layouts"""
    population = []
    
    # Add original layout
    population.append(base_layout)
    
    # Create random variations
    for _ in range(population_size - 1):
        individual = [dict(f) for f in base_layout]
        
        # Randomize positions
        for furniture in individual:
            w, h = planner.get_furniture_dimensions(furniture)
            
            # Try random positions
            for _ in range(50):
                furniture['x'] = random.uniform(0, ROOM_WIDTH - w)
                furniture['y'] = random.uniform(0, ROOM_HEIGHT - h)
                furniture['rotation'] = random.choice([0, 90, 180, 270])
                
                if planner.is_position_valid(furniture, individual):
                    break
        
        population.append(individual)
    
    return population

def genetic_algorithm(initial_layout, generations=80, population_size=30):
    """Run genetic algorithm optimization"""
    planner = BarrierFreePlanner()
    population = create_initial_population(initial_layout, population_size, planner)
    
    best_scores = []
    
    for gen in range(generations):
        # Evaluate and sort population
        population.sort(key=planner.evaluate_layout, reverse=True)
        current_best_score = planner.evaluate_layout(population[0])
        best_scores.append(current_best_score)
        
        print(f"Generation {gen}: Best fitness = {current_best_score:.2f}")
        
        # Selection and reproduction
        elite_size = max(2, population_size // 8)
        next_gen = population[:elite_size]  # Keep best individuals
        
        while len(next_gen) < population_size:
            # Tournament selection
            parent1 = max(random.choices(population[:population_size//2], k=3), 
                         key=planner.evaluate_layout)
            parent2 = max(random.choices(population[:population_size//2], k=3), 
                         key=planner.evaluate_layout)
            
            # Crossover
            child = planner.crossover(parent1, parent2)
            
            # Mutation
            if random.random() < 0.7:
                child = planner.mutate_layout(child, mutation_rate=0.2)
            
            next_gen.append(child)
        
        population = next_gen
        
        # Early stopping if solution is very good
        if current_best_score > 2500:
            print(f"Excellent solution found at generation {gen}")
            break
    
    return population[0], best_scores, planner

def visualize_layout(before_layout, after_layout, planner):
    """Visualize before and after layouts"""
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    titles = ["Original Layout", "Optimized Layout"]
    layouts = [before_layout, after_layout]
    
    for i in range(2):
        ax = axs[i]
        ax.set_xlim(0, ROOM_WIDTH)
        ax.set_ylim(0, ROOM_HEIGHT)
        ax.set_title(f"{titles[i]} (Score: {planner.evaluate_layout(layouts[i]):.1f})")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        layout = layouts[i]
        
        # Draw clearance zones
        for furniture in layout:
            clearance_zones = planner.create_clearance_zones(furniture)
            for zone in clearance_zones:
                color = 'orange' if zone["type"] == "front" else 'yellow'
                alpha = 0.2 if zone["type"] == "front" else 0.1
                clr_rect = patches.Rectangle(
                    (zone["x"], zone["y"]), zone["width"], zone["height"],
                    linewidth=1, edgecolor=color, facecolor=color,
                    alpha=alpha, linestyle='--')
                ax.add_patch(clr_rect)
        
        # Draw furniture
        for furniture in layout:
            w, h = planner.get_furniture_dimensions(furniture)
            
            rect = patches.Rectangle((furniture["x"], furniture["y"]), w, h,
                                   linewidth=2, edgecolor='black',
                                   facecolor='lightblue', alpha=0.8)
            ax.add_patch(rect)
            
            # Furniture name and accessibility indicator
            center_x, center_y = furniture["x"] + w/2, furniture["y"] + h/2
            ax.text(center_x, center_y, furniture["name"], ha='center', va='center',
                   fontsize=9, weight='bold')
            
            # Accessibility indicator
            if planner.has_wheelchair_access(furniture, layout):
                ax.plot(center_x, center_y, 'g*', markersize=10, alpha=0.8)
            else:
                ax.plot(center_x, center_y, 'r*', markersize=8, alpha=0.8)
        
        # Draw openings
        for opening in openings:
            door_rect = patches.Rectangle(
                (opening["x"], opening["y"]), opening["width"], opening["height"],
                linewidth=3, edgecolor='green', facecolor='lightgreen', alpha=0.6)
            ax.add_patch(door_rect)
            ax.text(opening["x"] + opening["width"]/2, opening["y"] + opening["height"]/2,
                   'DOOR', ha='center', va='center', fontweight='bold', color='darkgreen')
        
        ax.set_xlabel("Width (cm)")
        ax.set_ylabel("Height (cm)")
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lightblue', label='Furniture'),
        patches.Patch(color='orange', alpha=0.3, label='Front Clearance'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='g', markersize=10, label='Wheelchair Accessible'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='r', markersize=8, label='Limited Access'),
        patches.Patch(color='lightgreen', label='Door/Opening')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=5)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()

def analyze_layout(layout, planner, title="Layout Analysis"):
    """Analyze and print layout metrics"""
    print(f"\n{title}")
    print("="*50)
    
    total_score = planner.evaluate_layout(layout)
    print(f"Overall Score: {total_score:.2f}")
    
    # Accessibility analysis
    accessible_count = sum(1 for f in layout if planner.has_wheelchair_access(f, layout))
    accessibility_percentage = (accessible_count / len(layout)) * 100
    print(f"Wheelchair Accessible Furniture: {accessible_count}/{len(layout)} ({accessibility_percentage:.1f}%)")
    
    # Space utilization
    total_furniture_area = sum(
        planner.get_furniture_dimensions(f)[0] * planner.get_furniture_dimensions(f)[1] 
        for f in layout
    )
    room_area = ROOM_WIDTH * ROOM_HEIGHT
    utilization = (total_furniture_area / room_area) * 100
    print(f"Space Utilization: {utilization:.1f}%")
    
    # Relationship quality
    relationship_score = planner.evaluate_relationships(layout)
    print(f"Relationship Quality: {relationship_score:.1f}")

def save_layout(layout, filename="optimized_layout.json"):
    """Save the optimized layout"""
    output_data = {
        "room": {"width": ROOM_WIDTH, "height": ROOM_HEIGHT},
        "furniture": layout,
        "openings": openings
    }
    
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Layout saved to {filename}")

def load_initial_layout():
    """Load the original furniture layout"""
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

# Main execution
if __name__ == "__main__":
    print("🏠 Starting Barrier-Free Furniture Layout Optimization...")
    print(f"Room dimensions: {ROOM_WIDTH}cm x {ROOM_HEIGHT}cm")
    print(f"Number of furniture items: {len(furniture_items)}")
    
    # Load initial layout
    initial_layout = load_initial_layout()
    
    # Analyze original layout
    planner = BarrierFreePlanner()
    analyze_layout(initial_layout, planner, "Original Layout Analysis")
    
    # Run genetic algorithm
    print("\n🧬 Running genetic algorithm optimization...")
    best_layout, fitness_evolution, planner = genetic_algorithm(
        initial_layout, 
        generations=100, 
        population_size=40
    )
    
    # Analyze optimized layout
    analyze_layout(best_layout, planner, "Optimized Layout Analysis")
    
    # Save results
    save_layout(best_layout)
    
    # Visualize results
    print("\n🎨 Generating visualization...")
    visualize_layout(initial_layout, best_layout, planner)
    
    # Plot fitness evolution
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_evolution, linewidth=2)
    plt.title('Fitness Evolution Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n✅ Optimization complete!")