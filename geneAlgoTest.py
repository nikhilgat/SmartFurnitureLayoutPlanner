import math
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import copy

# Load config JSON
with open("room-layout-new.json", "r") as f:
    config = json.load(f)

# Load barrier-free constraints
with open("barrier_free_constraints_relational.json", "r") as f:
    bf_constraints = json.load(f)

ROOM_WIDTH = config["room"]["width"]
ROOM_HEIGHT = config["room"]["height"]
furniture_items = config["furniture"]
openings = config.get("openings", [])

class EnhancedDeterministicBarrierFreePlanner:
    def __init__(self): 
        self.clearance_reqs = bf_constraints["clearance_requirements"]
        self.furniture_clearances = bf_constraints["furniture_specific_clearances"]
        self.relationships = bf_constraints["furniture_relationships"]
        self.ergonomics = bf_constraints["ergonomic_constraints"]
        self.accessibility = bf_constraints["accessibility_enhancements"]
        
        # Define furniture groups and their relationships
        self.furniture_groups = self._create_furniture_groups()
        
        # Define group priority order (most important first)
        self.group_priority = [
            "sleeping_zone", "bathroom_zone", "storage_zone", 
            "dining_zone", "living_zone", "work_zone"
        ]
        
        # Define individual furniture priority within groups
        self.furniture_priority = [
            "Bed", "Toilet", "Washbasin", "Wardrobe", "Dining Table", 
            "Desk", "Sofa", "Chair", "Coffee Table", "TV Cabinet"
        ]
        
        # Define preferred wall positions for different furniture types
        self.wall_preferences = {
            "Bed": ["against_wall"],
            "Wardrobe": ["against_wall"], 
            "Sofa": ["against_wall", "corner"],
            "Desk": ["against_wall"],
            "TV Cabinet": ["against_wall"],
            "Toilet": ["corner"],
            "Washbasin": ["against_wall"]
        }
    
    def _create_furniture_groups(self) -> Dict:
        """Create functional furniture groups based on relationships"""
        groups = {
            "sleeping_zone": {
                "primary": ["Bed", "King Bed", "Queen Bed", "Twin Bed"],
                "secondary": ["Bedside Table", "Nightstand"],
                "optional": ["Wardrobe", "Dresser", "Armchair"]
            },
            "dining_zone": {
                "primary": ["Dining Table", "Kitchen Table"],
                "secondary": ["Dining Chair", "Chair"],
                "optional": ["Sideboard", "China Cabinet"]
            },
            "living_zone": {
                "primary": ["Sofa", "Sectional", "Loveseat"],
                "secondary": ["Coffee Table", "Side Table"],
                "optional": ["TV Cabinet", "Armchair", "Floor Lamp"]
            },
            "work_zone": {
                "primary": ["Desk", "Office Desk", "Writing Desk"],
                "secondary": ["Office Chair", "Desk Chair"],
                "optional": ["Bookshelf", "Filing Cabinet"]
            },
            "bathroom_zone": {
                "primary": ["Toilet", "Washbasin"],
                "secondary": ["Shower", "Bathtub"],
                "optional": []
            },
            "storage_zone": {
                "primary": ["Wardrobe", "Closet", "Armoire"],
                "secondary": ["Dresser", "Chest of Drawers"],
                "optional": ["Bench"]
            }
        }
        return groups
    
    def normalize_furniture_name(self, name: str) -> str:
        """Normalize furniture names for constraint lookup"""
        name_map = {
            "bed": "Bed", "sofa": "Sofa", "chair": "Chair", "dining chair": "Dining Chair",
            "table": "Dining Table", "dining table": "Dining Table", "coffee table": "Coffee Table",
            "desk": "Desk", "wardrobe": "Wardrobe", "tv cabinet": "TV Cabinet",
            "sink": "Washbasin", "toilet": "Toilet", "armchair": "Armchair",
            "nightstand": "Bedside Table", "bedside table": "Bedside Table"
        }
        return name_map.get(name.lower(), name.title())
    
    def get_furniture_dimensions(self, furniture: Dict) -> Tuple[float, float]:
        """Get actual dimensions considering rotation"""
        if furniture["rotation"] in [90, 270]:
            return furniture["height"], furniture["width"]
        return furniture["width"], furniture["height"]
    
    def identify_furniture_group(self, furniture: Dict) -> str:
        """Identify which functional group a furniture item belongs to"""
        norm_name = self.normalize_furniture_name(furniture["name"])
        
        for group_name, group_info in self.furniture_groups.items():
            all_furniture = group_info["primary"] + group_info["secondary"] + group_info["optional"]
            if norm_name in all_furniture:
                return group_name
        
        return "miscellaneous"
    
    def get_furniture_partners(self, furniture: Dict) -> List[str]:
        """Get required partners for a furniture item"""
        norm_name = self.normalize_furniture_name(furniture["name"])
        functional_pairs = self.relationships["functional_pairs"]
        
        if norm_name in functional_pairs:
            required_partners = functional_pairs[norm_name].get("required_partners", [])
            optional_partners = functional_pairs[norm_name].get("optional_partners", [])
            return required_partners + optional_partners
        
        return []
    
    def find_partner_furniture(self, furniture: Dict, furniture_list: List[Dict]) -> List[Dict]:
        """Find partner furniture items that should be placed together"""
        partners = []
        partner_names = self.get_furniture_partners(furniture)
        
        for other_furniture in furniture_list:
            other_norm_name = self.normalize_furniture_name(other_furniture["name"])
            if other_norm_name in partner_names and other_furniture != furniture:
                partners.append(other_furniture)
        
        return partners
    
    def create_clearance_zones(self, furniture: Dict) -> List[Dict]:
        """Create accessibility clearance zones around furniture"""
        w, h = self.get_furniture_dimensions(furniture)
        norm_name = self.normalize_furniture_name(furniture["name"])
        
        if norm_name not in self.furniture_clearances:
            return [{
                "x": furniture["x"] - 45, "y": furniture["y"] - 45,
                "width": w + 90, "height": h + 90, "type": "general"
            }]
        
        clearance = self.furniture_clearances[norm_name]
        zones = []
        front_clear = clearance.get("front_clearance", 90)
        
        # Create front clearance zone based on rotation
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
    
    def check_rectangle_overlap(self, rect1: Dict, rect2: Dict) -> bool:
        """Check if two rectangles overlap"""
        return not (rect1["x"] + rect1["width"] <= rect2["x"] or
                   rect1["x"] >= rect2["x"] + rect2["width"] or
                   rect1["y"] + rect1["height"] <= rect2["y"] or
                   rect1["y"] >= rect2["y"] + rect2["height"])
    
    def get_opening_rect(self, opening: Dict) -> Dict:
        """Convert wall-based opening to rectangle coords, or return as-is if already in rect format."""
        if "x" in opening and "y" in opening:
            return opening  # Already in rect format
        
        if "wall" not in opening:
            raise ValueError(f"Invalid opening format: {opening}")
        
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
        else:
            raise ValueError(f"Unknown wall: {wall}")
    
    def is_position_valid(self, furniture: Dict, layout: List[Dict]) -> bool:
        """Check if furniture position is valid"""
        w, h = self.get_furniture_dimensions(furniture)
        
        # Check room boundaries
        if not (0 <= furniture['x'] <= ROOM_WIDTH - w and 0 <= furniture['y'] <= ROOM_HEIGHT - h):
            return False
        
        furniture_rect = {'x': furniture['x'], 'y': furniture['y'], 'width': w, 'height': h}
        
        # Check overlaps with existing furniture and their clearances
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
            opening_rect = self.get_opening_rect(opening)
            if self.check_rectangle_overlap(furniture_rect, opening_rect):
                return False
        
        return True
    
    def get_partner_positions(self, primary_furniture: Dict, partner: Dict) -> List[Dict]:
        """Get optimal positions for partner furniture relative to primary furniture"""
        primary_norm = self.normalize_furniture_name(primary_furniture["name"])
        partner_norm = self.normalize_furniture_name(partner["name"])
        
        functional_pairs = self.relationships["functional_pairs"]
        positions = []
        
        if primary_norm in functional_pairs:
            constraints = functional_pairs[primary_norm]
            orientation = constraints.get("orientation", "adjacent")
            min_dist = constraints.get("min_distance", 30)
            max_dist = constraints.get("max_distance", 150)
            optimal_dist = (min_dist + max_dist) / 2
            
            primary_w, primary_h = self.get_furniture_dimensions(primary_furniture)
            partner_w, partner_h = self.get_furniture_dimensions(partner)
            
            # Generate positions based on relationship type
            if orientation == "facing":
                positions.extend(self._get_facing_positions(
                    primary_furniture, partner, optimal_dist, primary_w, primary_h, partner_w, partner_h
                ))
            elif orientation == "adjacent":
                positions.extend(self._get_adjacent_positions(
                    primary_furniture, partner, optimal_dist, primary_w, primary_h, partner_w, partner_h
                ))
            elif orientation == "parallel":
                positions.extend(self._get_parallel_positions(
                    primary_furniture, partner, optimal_dist, primary_w, primary_h, partner_w, partner_h
                ))
        
        return positions
    
    def _get_facing_positions(self, primary: Dict, partner: Dict, dist: float, pw: float, ph: float, sw: float, sh: float) -> List[Dict]:
        """Get positions where partner faces the primary furniture"""
        positions = []
        px, py = primary["x"], primary["y"]
        
        close_dist = min(dist, 80)
        positions.append({"x": px + (pw - sw) // 2, "y": py + ph + close_dist, "rotation": 180})
        positions.append({"x": px + (pw - sw) // 2, "y": py - close_dist - sh, "rotation": 0})
        positions.append({"x": px + pw + close_dist, "y": py + (ph - sh) // 2, "rotation": 270})
        positions.append({"x": px - close_dist - sw, "y": py + (ph - sh) // 2, "rotation": 90})
        
        return positions
    
    def _get_adjacent_positions(self, primary: Dict, partner: Dict, dist: float, pw: float, ph: float, sw: float, sh: float) -> List[Dict]:
        """Get positions where partner is adjacent to primary furniture"""
        positions = []
        px, py = primary["x"], primary["y"]
        
        close_dist = min(dist, 60)
        positions.append({"x": px + pw + close_dist, "y": py, "rotation": 0})
        positions.append({"x": px + pw + close_dist, "y": py + ph - sh, "rotation": 0})
        positions.append({"x": px - close_dist - sw, "y": py, "rotation": 0})
        positions.append({"x": px - close_dist - sw, "y": py + ph - sh, "rotation": 0})
        positions.append({"x": px, "y": py + ph + close_dist, "rotation": 0})
        positions.append({"x": px + pw - sw, "y": py + ph + close_dist, "rotation": 0})
        positions.append({"x": px, "y": py - close_dist - sh, "rotation": 0})
        positions.append({"x": px + pw - sw, "y": py - close_dist - sh, "rotation": 0})
        
        return positions
    
    def _get_parallel_positions(self, primary: Dict, partner: Dict, dist: float, pw: float, ph: float, sw: float, sh: float) -> List[Dict]:
        """Get positions where partner is parallel to primary furniture"""
        positions = []
        px, py = primary["x"], primary["y"]
        
        positions.append({"x": px, "y": py + ph + dist, "rotation": 0})
        positions.append({"x": px + (pw - sw), "y": py + ph + dist, "rotation": 0})
        positions.append({"x": px, "y": py - dist - sh, "rotation": 0})
        positions.append({"x": px + (pw - sw), "y": py - dist - sh, "rotation": 0})
        
        return positions
    
    def get_wall_positions(self, furniture: Dict) -> List[Dict]:
        """Get deterministic wall positions for furniture"""
        w, h = self.get_furniture_dimensions(furniture)
        positions = []
        
        wall_positions = [
            {"x": 0, "y": ROOM_HEIGHT//4, "rotation": 0},
            {"x": 0, "y": ROOM_HEIGHT//2, "rotation": 0},
            {"x": 0, "y": ROOM_HEIGHT*3//4 - h, "rotation": 0},
            {"x": ROOM_WIDTH - w, "y": ROOM_HEIGHT//4, "rotation": 0},
            {"x": ROOM_WIDTH - w, "y": ROOM_HEIGHT//2, "rotation": 0},
            {"x": ROOM_WIDTH - w, "y": ROOM_HEIGHT*3//4 - h, "rotation": 0},
            {"x": ROOM_WIDTH//4, "y": 0, "rotation": 0},
            {"x": ROOM_WIDTH//2 - w//2, "y": 0, "rotation": 0},
            {"x": ROOM_WIDTH*3//4 - w, "y": 0, "rotation": 0},
            {"x": ROOM_WIDTH//4, "y": ROOM_HEIGHT - h, "rotation": 0},
            {"x": ROOM_WIDTH//2 - w//2, "y": ROOM_HEIGHT - h, "rotation": 0},
            {"x": ROOM_WIDTH*3//4 - w, "y": ROOM_HEIGHT - h, "rotation": 0},
        ]
        
        corner_positions = [
            {"x": 0, "y": 0, "rotation": 0},
            {"x": ROOM_WIDTH - w, "y": 0, "rotation": 0},
            {"x": 0, "y": ROOM_HEIGHT - h, "rotation": 0},
            {"x": ROOM_WIDTH - w, "y": ROOM_HEIGHT - h, "rotation": 0},
        ]
        
        norm_name = self.normalize_furniture_name(furniture["name"])
        preferences = self.wall_preferences.get(norm_name, [])
        
        if "against_wall" in preferences:
            positions.extend(wall_positions)
        if "corner" in preferences:
            positions.extend(corner_positions)
        
        if not positions:
            positions = wall_positions + corner_positions
            
        return positions
    
    def score_position(self, furniture: Dict, layout: List[Dict]) -> float:
        """Score a furniture position based on multiple criteria"""
        score = 0
        norm_name = self.normalize_furniture_name(furniture["name"])
        w, h = self.get_furniture_dimensions(furniture)
        
        # Wall proximity bonus
        wall_bonus = 0
        if furniture['x'] <= 10:
            wall_bonus += 30
        if furniture['x'] + w >= ROOM_WIDTH - 10:
            wall_bonus += 30
        if furniture['y'] <= 10:
            wall_bonus += 25
        if furniture['y'] + h >= ROOM_HEIGHT - 10:
            wall_bonus += 25
        score += wall_bonus
        
        # Corner bonus
        corner_bonus = 0
        if ((furniture['x'] <= 10 and furniture['y'] <= 10) or
            (furniture['x'] + w >= ROOM_WIDTH - 10 and furniture['y'] <= 10) or
            (furniture['x'] <= 10 and furniture['y'] + h >= ROOM_HEIGHT - 10) or
            (furniture['x'] + w >= ROOM_WIDTH - 10 and furniture['y'] + h >= ROOM_HEIGHT - 10)):
            if norm_name in ['Toilet', 'Wardrobe', 'Desk']:
                corner_bonus = 40
        score += corner_bonus
        
        # Center avoidance for large furniture
        if norm_name in ['Bed', 'Sofa', 'Wardrobe']:
            center_x = ROOM_WIDTH / 2
            center_y = ROOM_HEIGHT / 2
            furniture_center_x = furniture['x'] + w / 2
            furniture_center_y = furniture['y'] + h / 2
            distance_from_center = math.hypot(furniture_center_x - center_x, furniture_center_y - center_y)
            center_avoidance_bonus = min(50, distance_from_center / 10)
            score += center_avoidance_bonus
        
        # Relationship bonus
        relationship_bonus = 0
        functional_pairs = self.relationships["functional_pairs"]
        
        if norm_name in functional_pairs:
            constraints = functional_pairs[norm_name]
            for partner_type in constraints.get("required_partners", []):
                partners = [f for f in layout if self.normalize_furniture_name(f["name"]) == partner_type]
                if partners:
                    closest_partner = min(partners, key=lambda p: self.calculate_distance(furniture, p))
                    distance = self.calculate_distance(furniture, closest_partner)
                    min_dist = constraints.get("min_distance", 30)
                    max_dist = constraints.get("max_distance", 200)
                    optimal_dist = (min_dist + max_dist) / 2
                    if min_dist <= distance <= max_dist:
                        distance_score = 150 - abs(distance - optimal_dist) / optimal_dist * 75
                        relationship_bonus += max(75, distance_score)
                    else:
                        relationship_bonus -= 50
            score += relationship_bonus
        
        return score
    
    def calculate_distance(self, furniture1: Dict, furniture2: Dict) -> float:
        """Calculate distance between two furniture items (center-to-center)"""
        w1, h1 = self.get_furniture_dimensions(furniture1)
        w2, h2 = self.get_furniture_dimensions(furniture2)
        center1_x = furniture1["x"] + w1 / 2
        center1_y = furniture1["y"] + h1 / 2
        center2_x = furniture2["x"] + w2 / 2
        center2_y = furniture2["y"] + h2 / 2
        return math.hypot(center1_x - center2_x, center1_y - center2_y)
    
    def has_wheelchair_access(self, furniture: Dict, layout: List[Dict]) -> bool:
        """Check if furniture has sufficient wheelchair access"""
        clearance_zones = self.create_clearance_zones(furniture)
        for zone in clearance_zones:
            for other in layout:
                if other == furniture:
                    continue
                other_w, other_h = self.get_furniture_dimensions(other)
                other_rect = {"x": other["x"], "y": other["y"], "width": other_w, "height": other_h}
                if self.check_rectangle_overlap(zone, other_rect):
                    return False
        return True
    
    def evaluate_layout(self, layout: List[Dict]) -> float:
        """Evaluate the overall layout score"""
        score = 0
        for furniture in layout:
            score += self.score_position(furniture, layout)
        score += self.calculate_space_efficiency(layout)
        score += self.evaluate_circulation(layout)
        return score
    
    def calculate_space_efficiency(self, layout: List[Dict]) -> float:
        """Calculate space efficiency score"""
        total_furniture_area = sum(
            self.get_furniture_dimensions(f)[0] * self.get_furniture_dimensions(f)[1] 
            for f in layout
        )
        room_area = ROOM_WIDTH * ROOM_HEIGHT
        utilization_ratio = total_furniture_area / room_area
        if 0.25 <= utilization_ratio <= 0.4:
            return 100
        elif utilization_ratio < 0.25:
            return 50
        else:
            return max(0, 100 - (utilization_ratio - 0.4) * 200)
    
    def evaluate_circulation(self, layout: List[Dict]) -> float:
        """Evaluate circulation space"""
        center_rect = {
            'x': ROOM_WIDTH * 0.3,
            'y': ROOM_HEIGHT * 0.3,
            'width': ROOM_WIDTH * 0.4,
            'height': ROOM_HEIGHT * 0.4
        }
        blocking_count = 0
        for furniture in layout:
            w, h = self.get_furniture_dimensions(furniture)
            furniture_rect = {'x': furniture['x'], 'y': furniture['y'], 'width': w, 'height': h}
            if self.check_rectangle_overlap(center_rect, furniture_rect):
                blocking_count += 1
        return max(0, 100 - blocking_count * 25)
    
    def analyze_relationships(self, layout: List[Dict]) -> Dict:
        """Analyze relationship compliance in the layout"""
        analysis = {
            "functional_groups": {},
            "missing_relationships": [],
            "optimal_relationships": [],
            "suboptimal_relationships": []
        }
        
        for group_name, group_info in self.furniture_groups.items():
            group_furniture = []
            for furniture in layout:
                if self.identify_furniture_group(furniture) == group_name:
                    group_furniture.append(furniture)
            if group_furniture:
                analysis["functional_groups"][group_name] = {
                    "furniture_count": len(group_furniture),
                    "furniture_items": [f["name"] for f in group_furniture]
                }
        
        functional_pairs = self.relationships["functional_pairs"]
        for primary_name, constraints in functional_pairs.items():
            primary_items = [f for f in layout if self.normalize_furniture_name(f["name"]) == primary_name]
            for primary_item in primary_items:
                for partner_type in constraints.get("required_partners", []):
                    partner_items = [f for f in layout if self.normalize_furniture_name(f["name"]) == partner_type]
                    if not partner_items:
                        analysis["missing_relationships"].append(f"{primary_name} needs {partner_type}")
                    else:
                        closest_partner = min(partner_items, key=lambda p: self.calculate_distance(primary_item, p))
                        distance = self.calculate_distance(primary_item, closest_partner)
                        min_dist = constraints.get("min_distance", 30)
                        max_dist = constraints.get("max_distance", 200)
                        if min_dist <= distance <= max_dist:
                            analysis["optimal_relationships"].append(
                                f"{primary_name} ↔ {partner_type}: {distance:.1f}cm (optimal)"
                            )
                        else:
                            analysis["suboptimal_relationships"].append(
                                f"{primary_name} ↔ {partner_type}: {distance:.1f}cm (should be {min_dist}-{max_dist}cm)"
                            )
        
        return analysis
    
    def find_optimal_position_with_partners(self, furniture: Dict, partners: List[Dict], layout: List[Dict]) -> Dict:
        """Find optimal position for furniture considering partners"""
        best_position = None
        best_score = float('-inf')
        
        for pos in self.get_wall_positions(furniture):
            test_furniture = copy.deepcopy(furniture)
            test_furniture.update(pos)
            if self.is_position_valid(test_furniture, layout):
                score = self.score_position(test_furniture, layout)
                if score > best_score:
                    best_score = score
                    best_position = test_furniture
        
        return best_position if best_position else furniture
    
    def create_relationship_aware_layout(self, furniture_list: List[Dict]) -> List[Dict]:
        """Create a relationship-aware layout"""
        optimized_layout = []
        remaining_furniture = copy.deepcopy(furniture_list)
        
        # Sort furniture by priority
        sorted_furniture = sorted(
            remaining_furniture,
            key=lambda f: (
                self.group_priority.index(self.identify_furniture_group(f)) if self.identify_furniture_group(f) in self.group_priority else len(self.group_priority),
                self.furniture_priority.index(self.normalize_furniture_name(f["name"])) if self.normalize_furniture_name(f["name"]) in self.furniture_priority else len(self.furniture_priority)
            )
        )
        
        while sorted_furniture:
            furniture = sorted_furniture.pop(0)
            partners = self.find_partner_furniture(furniture, sorted_furniture)
            
            # Place primary furniture
            placed_furniture = self.find_optimal_position_with_partners(furniture, partners, optimized_layout)
            if placed_furniture:
                optimized_layout.append(placed_furniture)
            
            # Place partners
            for partner in partners:
                partner_positions = self.get_partner_positions(placed_furniture, partner)
                best_partner_pos = None
                best_score = float('-inf')
                
                for pos in partner_positions:
                    test_partner = copy.deepcopy(partner)
                    test_partner.update(pos)
                    if self.is_position_valid(test_partner, optimized_layout):
                        score = self.score_position(test_partner, optimized_layout + [placed_furniture])
                        if score > best_score:
                            best_score = score
                            best_partner_pos = test_partner
                
                if best_partner_pos:
                    optimized_layout.append(best_partner_pos)
                    sorted_furniture.remove(partner)
        
        return optimized_layout

def load_initial_layout() -> List[Dict]:
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

def create_relationship_aware_layout(initial_layout: List[Dict]) -> Tuple[List[Dict], object]:
    """Create a relationship-aware, consistent layout using enhanced deterministic algorithm"""
    planner = EnhancedDeterministicBarrierFreePlanner()
    furniture_copy = copy.deepcopy(initial_layout)
    optimized_layout = planner.create_relationship_aware_layout(furniture_copy)
    return optimized_layout, planner

def visualize_layout(before_layout: List[Dict], after_layout: List[Dict], planner):
    """Visualize before and after layouts with relationship indicators"""
    fig, axs = plt.subplots(1, 2, figsize=(22, 11))
    titles = ["Original Layout", "Relationship-Optimized Layout"]
    layouts = [before_layout, after_layout]
    
    for i in range(2):
        ax = axs[i]
        ax.set_xlim(0, ROOM_WIDTH)
        ax.set_ylim(0, ROOM_HEIGHT)
        ax.set_title(f"{titles[i]} (Score: {planner.evaluate_layout(layouts[i]):.1f})")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        layout = layouts[i]
        
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
        
        functional_pairs = planner.relationships["functional_pairs"]
        drawn_connections = set()
        
        for furniture in layout:
            furniture_norm = planner.normalize_furniture_name(furniture["name"])
            if furniture_norm in functional_pairs:
                constraints = functional_pairs[furniture_norm]
                all_partners = constraints.get("required_partners", []) + constraints.get("optional_partners", [])
                for partner_type in all_partners:
                    partner_items = [f for f in layout if planner.normalize_furniture_name(f["name"]) == partner_type]
                    if partner_items:
                        closest_partner = min(partner_items, key=lambda p: planner.calculate_distance(furniture, p))
                        connection_key = tuple(sorted([furniture["name"], closest_partner["name"]]))
                        if connection_key not in drawn_connections:
                            drawn_connections.add(connection_key)
                            fw, fh = planner.get_furniture_dimensions(furniture)
                            pw, ph = planner.get_furniture_dimensions(closest_partner)
                            f_center = (furniture["x"] + fw/2, furniture["y"] + fh/2)
                            p_center = (closest_partner["x"] + pw/2, closest_partner["y"] + ph/2)
                            distance = planner.calculate_distance(furniture, closest_partner)
                            min_dist = constraints.get("min_distance", 30)
                            max_dist = constraints.get("max_distance", 200)
                            is_required = partner_type in constraints.get("required_partners", [])
                            line_color = 'darkgreen' if is_required and min_dist <= distance <= max_dist else 'green' if min_dist <= distance <= max_dist else 'darkred' if is_required else 'red'
                            line_style = '-' if min_dist <= distance <= max_dist else '--'
                            alpha = 0.8 if is_required else 0.6
                            linewidth = 3 if is_required else 2
                            ax.plot([f_center[0], p_center[0]], [f_center[1], p_center[1]], 
                                   color=line_color, linestyle=line_style, linewidth=linewidth, alpha=alpha)
                            mid_x = (f_center[0] + p_center[0]) / 2
                            mid_y = (f_center[1] + p_center[1]) / 2
                            ax.text(mid_x, mid_y, f'{distance:.0f}cm', 
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
                                   fontsize=8, ha='center', va='center')
        
        for furniture in layout:
            w, h = planner.get_furniture_dimensions(furniture)
            group = planner.identify_furniture_group(furniture)
            group_colors = {
                "sleeping_zone": 'lightblue',
                "dining_zone": 'lightcoral',
                "living_zone": 'lightgreen',
                "work_zone": 'lightyellow',
                "bathroom_zone": 'lightcyan',
                "storage_zone": 'plum',
                "miscellaneous": 'lightgray'
            }
            rect = patches.Rectangle((furniture["x"], furniture["y"]), w, h,
                                   linewidth=2, edgecolor='black',
                                   facecolor=group_colors.get(group, 'lightgray'), alpha=0.8)
            ax.add_patch(rect)
            center_x, center_y = furniture["x"] + w/2, furniture["y"] + h/2
            ax.text(center_x, center_y, furniture["name"], ha='center', va='center',
                   fontsize=8, weight='bold')
            ax.plot(center_x, center_y, 'g*' if planner.has_wheelchair_access(furniture, layout) else 'r*',
                   markersize=10 if planner.has_wheelchair_access(furniture, layout) else 8, alpha=0.8)
        
        for opening in openings:
            opening_rect = planner.get_opening_rect(opening)
            door_rect = patches.Rectangle(
                (opening_rect["x"], opening_rect["y"]), opening_rect["width"], opening_rect["height"],
                linewidth=3, edgecolor='green', facecolor='lightgreen', alpha=0.6)
            ax.add_patch(door_rect)
            ax.text(opening_rect["x"] + opening_rect["width"]/2, opening_rect["y"] + opening_rect["height"]/2,
                   'DOOR', ha='center', va='center', fontweight='bold', color='darkgreen')
        
        ax.set_xlabel("Width (cm)")
        ax.set_ylabel("Height (cm)")
    
    legend_elements = [
        plt.Line2D([0], [0], color='darkgreen', lw=3, label='Optimal Required Relationship'),
        plt.Line2D([0], [0], color='green', lw=2, label='Optimal Optional Relationship'),
        plt.Line2D([0], [0], color='darkred', lw=3, linestyle='--', label='Suboptimal Required'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Suboptimal Optional'),
        plt.Line2D([0], [0], marker='*', color='g', markersize=10, linestyle='None', label='Wheelchair Accessible'),
        plt.Line2D([0], [0], marker='*', color='r', markersize=8, linestyle='None', label='Limited Access')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
    
    plt.tight_layout()
    plt.show()

def analyze_layout(layout: List[Dict], planner, title: str = "Layout Analysis"):
    """Enhanced layout analysis including relationships"""
    print(f"\n{title}")
    print("="*70)
    
    total_score = planner.evaluate_layout(layout)
    print(f"Overall Score: {total_score:.2f}")
    
    accessible_count = sum(1 for f in layout if planner.has_wheelchair_access(f, layout))
    accessibility_percentage = (accessible_count / len(layout)) * 100 if layout else 0
    print(f"Wheelchair Accessible: {accessible_count}/{len(layout)} ({accessibility_percentage:.1f}%)")
    
    total_furniture_area = sum(
        planner.get_furniture_dimensions(f)[0] * planner.get_furniture_dimensions(f)[1] 
        for f in layout
    )
    room_area = ROOM_WIDTH * ROOM_HEIGHT
    utilization = (total_furniture_area / room_area) * 100
    print(f"Space Utilization: {utilization:.1f}%")
    
    rel_analysis = planner.analyze_relationships(layout)
    print(f"\n Functional Group Distribution:")
    for group, info in rel_analysis["functional_groups"].items():
        print(f"  • {group.replace('_', ' ').title()}: {info['furniture_count']} items")
        print(f"    Items: {', '.join(info['furniture_items'])}")
    
    print(f"\n Optimal Relationships ({len(rel_analysis['optimal_relationships'])}):")
    for rel in rel_analysis["optimal_relationships"]:
        print(f"  • {rel}")
    
    if rel_analysis["suboptimal_relationships"]:
        print(f"\n  Suboptimal Relationships ({len(rel_analysis['suboptimal_relationships'])}):")
        for rel in rel_analysis["suboptimal_relationships"]:
            print(f"  • {rel}")
    
    if rel_analysis["missing_relationships"]:
        print(f"\n Missing Relationships ({len(rel_analysis['missing_relationships'])}):")
        for rel in rel_analysis["missing_relationships"]:
            print(f"  • {rel}")
    
    print(f"\n Furniture Positions:")
    for furniture in layout:
        group = planner.identify_furniture_group(furniture)
        print(f"  • {furniture['name']} ({group}): ({furniture['x']:.0f}, {furniture['y']:.0f}) - {furniture['rotation']}°")

def save_layout(layout: List[Dict], filename: str = "relationship_optimized_layout.json"):
    """Save the relationship-optimized layout"""
    output_data = {
        "method": "Relationship-Aware Deterministic Layout",
        "room": {"width": ROOM_WIDTH, "height": ROOM_HEIGHT},
        "furniture": layout,
        "openings": openings,
        "features": [
            "Functional relationship optimization",
            "Wheelchair accessibility compliance", 
            "Deterministic and repeatable results",
            "Zone-based furniture grouping",
            "Partner-aware placement"
        ],
        "note": "This layout prioritizes functional relationships between furniture items"
    }
    
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Relationship-optimized layout saved to {filename}")

if __name__ == "__main__":
    print(" Creating Relationship-Aware Deterministic Furniture Layout...")
    print(f"Room dimensions: {ROOM_WIDTH}cm x {ROOM_HEIGHT}cm")
    print(f"Number of furniture items: {len(furniture_items)}")
    print(" This method focuses on functional relationships between furniture!")
    
    initial_layout = load_initial_layout()
    print("\n Generating relationship-optimized layout...")
    optimized_layout, planner = create_relationship_aware_layout(initial_layout)
    
    analyze_layout(initial_layout, planner, "Original Layout Analysis")
    analyze_layout(optimized_layout, planner, "Relationship-Optimized Layout Analysis")
    
    save_layout(optimized_layout)
    
    print("\n Generating visualization with relationship indicators...")
    visualize_layout(initial_layout, optimized_layout, planner)
    
    print("\n Relationship-aware optimization complete!")
    print(" Run this script again - you'll get identical results every time!")
    print(" Key features:")
    print("   •  Functional relationship prioritization")
    print("   •  Tables placed with chairs")
    print("   •  Beds placed with nightstands") 
    print("   •  Desks placed with chairs")
    print("   •  Sofas placed with coffee tables")
    print("   •  Zone-based furniture grouping")
    print("   •  Consistent, repeatable results")
    print("   •  Wheelchair accessibility compliance")
    print("   •  Optimal distance relationships")
    print("\n Visualization Legend:")
    print("   • Green lines: Optimal relationships")
    print("   • Red dashed lines: Suboptimal relationships") 
    print("   • Green stars: Wheelchair accessible")
    print("   • Red stars: Limited accessibility")
    print("   • Color-coded furniture by functional zones")