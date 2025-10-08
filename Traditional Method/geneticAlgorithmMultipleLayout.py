import math
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import copy
import random


with open("Input-Layouts/room-layout-multiple.json", "r") as f:
    config = json.load(f)


with open("constraints/enhanced_barrier_free_constraints.json", "r") as f:
    bf_constraints = json.load(f)

ROOM_WIDTH = config["room"]["width"]
ROOM_HEIGHT = config["room"]["height"]
furniture_items = config["furniture"]
openings = config.get("openings", [])

class EnhancedDeterministicBarrierFreePlanner:
    def __init__(self, variation_seed=None): 
        self.clearance_reqs = bf_constraints["clearance_requirements"]
        self.furniture_clearances = bf_constraints["furniture_specific_clearances"]
        self.relationships = bf_constraints["furniture_relationships"]
        self.ergonomics = bf_constraints["ergonomic_constraints"]
        self.accessibility = bf_constraints["accessibility_enhancements"]
        
        self.furniture_groups = self._create_furniture_groups()
        
        self.group_priority = [
            "sleeping_zone", "bathroom_zone", "storage_zone", 
            "dining_zone", "living_zone", "work_zone"
        ]
        
        self.furniture_priority = [
            "Bed", "Toilet", "Washbasin", "Wardrobe", "Dining Table", 
            "Desk", "Sofa", "Chair", "Coffee Table", "TV Cabinet"
        ]
        
        self.wall_preferences = {
            "Bed": ["against_wall"],
            "Wardrobe": ["against_wall"], 
            "Sofa": ["against_wall"],
            "Desk": ["against_wall"],
            "TV Cabinet": ["against_wall"],
            "Toilet": ["corner"],
            "Washbasin": ["against_wall"]
        }
        
        self.door_clearance = 150  # Added for door maneuvering space
        self.variation_seed = variation_seed or 0
    
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
    
    def create_opening_clearance_zones(self) -> List[Dict]:
        """Create clearance zones for doors to prevent blocking"""
        zones = []
        extra = 30  # Extra width for latch side, etc.
        for opening in openings:
            if opening.get("type", "door").lower() != "door":
                continue  # No large clearance for windows to allow furniture in front
            opening_rect = self.get_opening_rect(opening)
            wall = opening["wall"].lower() if "wall" in opening else None
            if wall == "bottom":
                zone = {
                    "x": opening_rect["x"] - extra,
                    "y": opening_rect["y"],
                    "width": opening_rect["width"] + 2 * extra,
                    "height": self.door_clearance,
                    "type": "door_clearance"
                }
            elif wall == "top":
                zone = {
                    "x": opening_rect["x"] - extra,
                    "y": opening_rect["y"] - self.door_clearance + opening_rect["height"],
                    "width": opening_rect["width"] + 2 * extra,
                    "height": self.door_clearance,
                    "type": "door_clearance"
                }
            elif wall == "left":
                zone = {
                    "x": opening_rect["x"],
                    "y": opening_rect["y"] - extra,
                    "width": self.door_clearance,
                    "height": opening_rect["height"] + 2 * extra,
                    "type": "door_clearance"
                }
            elif wall == "right":
                zone = {
                    "x": opening_rect["x"] - self.door_clearance + opening_rect["width"],
                    "y": opening_rect["y"] - extra,
                    "width": self.door_clearance,
                    "height": opening_rect["height"] + 2 * extra,
                    "type": "door_clearance"
                }
            else:
                continue
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
        
        # Check overlaps with opening clearances (added to prevent blocking doors)
        opening_clearances = self.create_opening_clearance_zones()
        for zone in opening_clearances:
            if self.check_rectangle_overlap(furniture_rect, zone):
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
        positions = []
        px, py = primary["x"], primary["y"]
        close_dist = min(dist, 80)

        def is_valid_rotation(furniture_name, rotation):
            norm_name = self.normalize_furniture_name(furniture_name)
            if norm_name in ["Wardrobe", "Sofa"]:
                return True  # Validated in get_wall_positions
            return True

        candidate_positions = [
            {"x": px + (pw - sw) // 2, "y": py + ph + close_dist, "rotation": 180},
            {"x": px + (pw - sw) // 2, "y": py - close_dist - sh, "rotation": 0},
            {"x": px + pw + close_dist, "y": py + (ph - sh) // 2, "rotation": 270},
            {"x": px - close_dist - sw, "y": py + (ph - sh) // 2, "rotation": 90},
        ]

        for pos in candidate_positions:
            if is_valid_rotation(partner["name"], pos["rotation"]):
                positions.append(pos)

        return positions


    def _get_adjacent_positions(self, primary: Dict, partner: Dict, dist: float, pw: float, ph: float, sw: float, sh: float) -> List[Dict]:
        positions = []
        px, py = primary["x"], primary["y"]
        close_dist = min(dist, 60)

        def is_valid_rotation(furniture_name, rotation):
            norm_name = self.normalize_furniture_name(furniture_name)
            if norm_name in ["Wardrobe", "Sofa"]:
                return True  # Validated in get_wall_positions
            return True

        candidate_positions = [
            {"x": px + pw + close_dist, "y": py, "rotation": 0},
            {"x": px + pw + close_dist, "y": py + ph - sh, "rotation": 0},
            {"x": px - close_dist - sw, "y": py, "rotation": 0},
            {"x": px - close_dist - sw, "y": py + ph - sh, "rotation": 0},
            {"x": px, "y": py + ph + close_dist, "rotation": 0},
            {"x": px + pw - sw, "y": py + ph + close_dist, "rotation": 0},
            {"x": px, "y": py - close_dist - sh, "rotation": 0},
            {"x": px + pw - sw, "y": py - close_dist - sh, "rotation": 0},
        ]

        for pos in candidate_positions:
            if is_valid_rotation(partner["name"], pos["rotation"]):
                positions.append(pos)

        return positions

    
    def _get_parallel_positions(self, primary: Dict, partner: Dict, dist: float, pw: float, ph: float, sw: float, sh: float) -> List[Dict]:
        positions = []
        px, py = primary["x"], primary["y"]

        def is_valid_rotation(furniture_name, rotation):
            norm_name = self.normalize_furniture_name(furniture_name)
            if norm_name in ["Wardrobe", "Sofa"]:
                return True  # Validated in get_wall_positions
            return True

        candidate_positions = [
            {"x": px, "y": py + ph + dist, "rotation": 0},
            {"x": px + (pw - sw), "y": py + ph + dist, "rotation": 0},
            {"x": px, "y": py - dist - sh, "rotation": 0},
            {"x": px + (pw - sw), "y": py - dist - sh, "rotation": 0},
        ]

        for pos in candidate_positions:
            if is_valid_rotation(partner["name"], pos["rotation"]):
                positions.append(pos)

        return positions

    
    def get_wall_positions(self, furniture: Dict) -> List[Dict]:
        """Get deterministic wall positions for furniture, ensuring wardrobe and sofa face away from walls"""
        positions = []
        
        orig_w = furniture["width"]
        orig_h = furniture["height"]
        norm_name = self.normalize_furniture_name(furniture["name"])
        preferences = self.wall_preferences.get(norm_name, [])
        needs_face_away = norm_name in ["Wardrobe", "Sofa"]
        
        # Add variation based on seed for different layouts
        variation_factor = (self.variation_seed % 5) * 0.1  # 0.0 to 0.4
        
        if "against_wall" not in preferences:
            # Default positions with rotation 0, but vary placement slightly
            base_positions = [0.25, 0.5, 0.75]
            varied_positions = [pos + variation_factor * 0.1 for pos in base_positions]
            
            wall_positions = []
            for i, pos in enumerate(varied_positions):
                wall_positions.extend([
                    {"x": 0, "y": int(ROOM_HEIGHT * pos) - orig_h//2, "rotation": 0},
                    {"x": ROOM_WIDTH - orig_w, "y": int(ROOM_HEIGHT * pos) - orig_h//2, "rotation": 0},
                    {"x": int(ROOM_WIDTH * pos) - orig_w//2, "y": 0, "rotation": 0},
                    {"x": int(ROOM_WIDTH * pos) - orig_w//2, "y": ROOM_HEIGHT - orig_h, "rotation": 0},
                ])
            positions.extend(wall_positions)
        else:
            # Wall-specific rotations for facing away, with variation
            wall_configs = [
                ("left", 90),  # front right
                ("right", 270),  # front left
                ("bottom", 0),  # front top
                ("top", 180),  # front bottom
            ]
            
            base_positions = [0.25, 0.5, 0.75]
            varied_positions = [pos + variation_factor * 0.1 for pos in base_positions]
            
            for wall, rot in wall_configs:
                w = orig_h if rot in [90, 270] else orig_w
                h = orig_w if rot in [90, 270] else orig_h
                
                if wall == "left":
                    x = 0
                    for pos in varied_positions:
                        y = min(int(ROOM_HEIGHT * pos), ROOM_HEIGHT - h)
                        positions.append({"x": x, "y": y, "rotation": rot})
                elif wall == "right":
                    x = ROOM_WIDTH - w
                    for pos in varied_positions:
                        y = min(int(ROOM_HEIGHT * pos), ROOM_HEIGHT - h)
                        positions.append({"x": x, "y": y, "rotation": rot})
                elif wall == "bottom":
                    y = 0
                    for pos in varied_positions:
                        x = min(int(ROOM_WIDTH * pos), ROOM_WIDTH - w)
                        positions.append({"x": x, "y": y, "rotation": rot})
                elif wall == "top":
                    y = ROOM_HEIGHT - h
                    for pos in varied_positions:
                        x = min(int(ROOM_WIDTH * pos), ROOM_WIDTH - w)
                        positions.append({"x": x, "y": y, "rotation": rot})
        
        # Add corners if preferred, with variation
        if "corner" in preferences or needs_face_away:
            corner_configs = [
                ("bottom_left", [0, 90]),
                ("bottom_right", [0, 270]),
                ("top_left", [180, 90]),
                ("top_right", [180, 270]),
            ]
            
            # Vary corner placement slightly
            corner_offset = int(variation_factor * 50)  # Up to 20cm variation
            
            for corner, rots in corner_configs:
                for rot in rots:
                    w = orig_h if rot in [90, 270] else orig_w
                    h = orig_w if rot in [90, 270] else orig_h
                    if corner == "bottom_left":
                        x, y = corner_offset, corner_offset
                    elif corner == "bottom_right":
                        x, y = ROOM_WIDTH - w - corner_offset, corner_offset
                    elif corner == "top_left":
                        x, y = corner_offset, ROOM_HEIGHT - h - corner_offset
                    elif corner == "top_right":
                        x, y = ROOM_WIDTH - w - corner_offset, ROOM_HEIGHT - h - corner_offset
                    positions.append({"x": x, "y": y, "rotation": rot})
        
        if not positions:
            # Fallback with variation
            w = orig_w
            h = orig_h
            offset = int(variation_factor * 30)
            fallback_positions = [
                {"x": offset, "y": offset, "rotation": 0},
                {"x": ROOM_WIDTH - w - offset, "y": offset, "rotation": 0},
                {"x": offset, "y": ROOM_HEIGHT - h - offset, "rotation": 0},
                {"x": ROOM_WIDTH - w - offset, "y": ROOM_HEIGHT - h - offset, "rotation": 0},
            ]
            positions.extend(fallback_positions)
        
        return positions
    
    def score_position(self, furniture: Dict, layout: List[Dict]) -> float:
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
            corner_bonus += 20
        score += corner_bonus
        
        return score
    
    def calculate_distance(self, f1: Dict, f2: Dict) -> float:
        f1_w, f1_h = self.get_furniture_dimensions(f1)
        f2_w, f2_h = self.get_furniture_dimensions(f2)
        f1_center = (f1["x"] + f1_w / 2, f1["y"] + f1_h / 2)
        f2_center = (f2["x"] + f2_w / 2, f2["y"] + f2_h / 2)
        return math.hypot(f1_center[0] - f2_center[0], f1_center[1] - f2_center[1])
    
    def has_wheelchair_access(self, furniture: Dict, layout: List[Dict]) -> bool:
        clearance_zones = self.create_clearance_zones(furniture)
        for zone in clearance_zones:
            for existing in layout:
                if existing == furniture:
                    continue
                existing_rect = {
                    "x": existing["x"], "y": existing["y"],
                    "width": self.get_furniture_dimensions(existing)[0],
                    "height": self.get_furniture_dimensions(existing)[1]
                }
                if self.check_rectangle_overlap(zone, existing_rect):
                    return False
        return True
    
    def evaluate_layout(self, layout: List[Dict]) -> float:
        score = 0
        for furniture in layout:
            score += self.score_position(furniture, layout)
        return score
    
    def analyze_relationships(self, layout: List[Dict]) -> Dict:
        analysis = {
            "functional_groups": {},
            "optimal_relationships": [],
            "missing_relationships": [],
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
                                f"{primary_name} → {partner_type}: {distance:.1f}cm (optimal)"
                            )
                        else:
                            analysis["suboptimal_relationships"].append(
                                f"{primary_name} → {partner_type}: {distance:.1f}cm (should be {min_dist}-{max_dist}cm)"
                            )
        
        return analysis
    
    def find_optimal_position_with_partners(self, furniture: Dict, partners: List[Dict], layout: List[Dict]) -> Dict:
        """Find optimal position for furniture considering partners"""
        best_position = None
        best_score = float('-inf')
        
        # Get wall positions with variation based on seed
        wall_positions = self.get_wall_positions(furniture)
        
        # Add some randomization to position selection based on seed
        position_indices = list(range(len(wall_positions)))
        random.Random(self.variation_seed).shuffle(position_indices)
        
        for idx in position_indices:
            pos = wall_positions[idx]
            test_furniture = copy.deepcopy(furniture)
            test_furniture.update(pos)
            if self.is_position_valid(test_furniture, layout):
                score = self.score_position(test_furniture, layout)
                if score > best_score:
                    best_score = score
                    best_position = test_furniture
        
        return best_position if best_position else furniture
    
    def create_relationship_aware_layout(self, furniture_list: List[Dict]) -> List[Dict]:
        """Create a relationship-aware layout with variation based on seed"""
        optimized_layout = []
        remaining_furniture = copy.deepcopy(furniture_list)
        
        # Add variation to furniture processing order
        furniture_priority_varied = self.furniture_priority.copy()
        random.Random(self.variation_seed).shuffle(furniture_priority_varied)
        
        # Sort furniture by priority with variation
        sorted_furniture = sorted(
            remaining_furniture,
            key=lambda f: (
                self.group_priority.index(self.identify_furniture_group(f)) if self.identify_furniture_group(f) in self.group_priority else len(self.group_priority),
                furniture_priority_varied.index(self.normalize_furniture_name(f["name"])) if self.normalize_furniture_name(f["name"]) in furniture_priority_varied else len(furniture_priority_varied)
            )
        )
        
        while sorted_furniture:
            furniture = sorted_furniture.pop(0)
            partners = self.find_partner_furniture(furniture, sorted_furniture)
            
            # Place primary furniture
            placed_furniture = self.find_optimal_position_with_partners(furniture, partners, optimized_layout)
            if placed_furniture:
                optimized_layout.append(placed_furniture)
            
            # Place partners with variation in partner position selection
            for partner in partners:
                partner_positions = self.get_partner_positions(placed_furniture, partner)
                
                # Add randomization to partner position selection
                position_indices = list(range(len(partner_positions)))
                random.Random(self.variation_seed + hash(partner["name"])).shuffle(position_indices)
                
                best_partner_pos = None
                best_score = float('-inf')
                
                for idx in position_indices:
                    pos = partner_positions[idx]
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

def create_multiple_relationship_aware_layouts(initial_layout: List[Dict], num_layouts: int = 5) -> Tuple[List[List[Dict]], List[object]]:
    """Create multiple relationship-aware, consistent layouts using enhanced deterministic algorithm"""
    layouts = []
    planners = []
    
    for i in range(num_layouts):
        print(f"Generating layout {i+1}/{num_layouts}...")
        planner = EnhancedDeterministicBarrierFreePlanner(variation_seed=i)
        furniture_copy = copy.deepcopy(initial_layout)
        optimized_layout = planner.create_relationship_aware_layout(furniture_copy)
        layouts.append(optimized_layout)
        planners.append(planner)
    
    return layouts, planners

def visualize_multiple_layouts(layouts: List[List[Dict]], planners: List[object]):
    """Visualize all layouts in a grid"""
    num_layouts = len(layouts)
    cols = 3 if num_layouts >= 3 else num_layouts
    rows = (num_layouts + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(7*cols, 7*rows))
    if rows == 1 and cols == 1:
        axs = [axs]
    elif rows == 1 or cols == 1:
        axs = axs.flatten()
    else:
        axs = axs.flatten()
    
    for i in range(num_layouts):
        ax = axs[i]
        layout = layouts[i]
        planner = planners[i]
        
        ax.set_xlim(0, ROOM_WIDTH)
        ax.set_ylim(0, ROOM_HEIGHT)
        ax.set_title(f"Layout {i+1} (Score: {planner.evaluate_layout(layout):.1f})")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
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
        
        # Draw relationship lines
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
                            linewidth = 2 if is_required else 1.5
                            ax.plot([f_center[0], p_center[0]], [f_center[1], p_center[1]], 
                                   color=line_color, linestyle=line_style, linewidth=linewidth, alpha=alpha)
        
        # Draw furniture
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
                "miscellaneous": 'lightgray'}
            rect = patches.Rectangle((furniture["x"], furniture["y"]), w, h,
                                   linewidth=2, edgecolor='black',
                                   facecolor=group_colors.get(group, 'lightgray'), alpha=0.8)
            ax.add_patch(rect)
            center_x, center_y = furniture["x"] + w/2, furniture["y"] + h/2
            ax.text(center_x, center_y, furniture["name"], ha='center', va='center',
                   fontsize=7, weight='bold')
            ax.plot(center_x, center_y, 'g*' if planner.has_wheelchair_access(furniture, layout) else 'r*',
                   markersize=8 if planner.has_wheelchair_access(furniture, layout) else 6, alpha=0.8)
        
        # Draw openings
        for opening in openings:
            if opening.get("type", "door").lower() == "door":
                opening_rect = planner.get_opening_rect(opening)
                door_rect = patches.Rectangle(
                    (opening_rect["x"], opening_rect["y"]), opening_rect["width"], opening_rect["height"],
                    linewidth=3, edgecolor='green', facecolor='lightgreen', alpha=0.6)
                ax.add_patch(door_rect)
                ax.text(opening_rect["x"] + opening_rect["width"]/2, opening_rect["y"] + opening_rect["height"]/2,
                       'DOOR', ha='center', va='center', fontweight='bold', color='darkgreen', fontsize=6)
            elif opening.get("type", "window").lower() == "window":
                opening_rect = planner.get_opening_rect(opening)
                window_rect = patches.Rectangle(
                    (opening_rect["x"], opening_rect["y"]), opening_rect["width"], opening_rect["height"],
                    linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.6)
                ax.add_patch(window_rect)
                ax.text(opening_rect["x"] + opening_rect["width"]/2, opening_rect["y"] + opening_rect["height"]/2,
                       'WINDOW', ha='center', va='center', fontweight='bold', color='darkblue', fontsize=6)
        
        ax.set_xlabel("Width (cm)")
        ax.set_ylabel("Height (cm)")
    
    # Hide unused subplots
    for i in range(num_layouts, len(axs)):
        axs[i].set_visible(False)
    
    legend_elements = [
        plt.Line2D([0], [0], color='darkgreen', lw=2, label='Optimal Required Relationship'),
        plt.Line2D([0], [0], color='green', lw=1.5, label='Optimal Optional Relationship'),
        plt.Line2D([0], [0], color='darkred', lw=2, linestyle='--', label='Suboptimal Required'),
        plt.Line2D([0], [0], color='red', lw=1.5, linestyle='--', label='Suboptimal Optional'),
        plt.Line2D([0], [0], marker='*', color='g', markersize=8, linestyle='None', label='Wheelchair Accessible'),
        plt.Line2D([0], [0], marker='*', color='r', markersize=6, linestyle='None', label='Limited Access')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
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
    print(f"\n📍 Functional Group Distribution:")
    for group, info in rel_analysis["functional_groups"].items():
        print(f"  • {group.replace('_', ' ').title()}: {info['furniture_count']} items")
        print(f"    Items: {', '.join(info['furniture_items'])}")
    
    print(f"\n✅ Optimal Relationships ({len(rel_analysis['optimal_relationships'])}):")
    for rel in rel_analysis["optimal_relationships"]:
        print(f"  • {rel}")
    
    if rel_analysis["suboptimal_relationships"]:
        print(f"\n⚠️ Suboptimal Relationships ({len(rel_analysis['suboptimal_relationships'])}):")
        for rel in rel_analysis["suboptimal_relationships"]:
            print(f"  • {rel}")
    
    if rel_analysis["missing_relationships"]:
        print(f"\n❌ Missing Relationships ({len(rel_analysis['missing_relationships'])}):")
        for rel in rel_analysis["missing_relationships"]:
            print(f"  • {rel}")

def save_multiple_layouts(layouts: List[List[Dict]], filename_prefix: str = "relationship_optimized_layout"):
    """Save all relationship-optimized layouts"""
    for i, layout in enumerate(layouts):
        filename = f"{filename_prefix}_{i+1}.json"
        output_data = {
            "method": f"Relationship-Aware Layout Variation {i+1}",
            "room": {"width": ROOM_WIDTH, "height": ROOM_HEIGHT},
            "furniture": layout,
            "openings": openings,
            "features": [
                "Functional relationship optimization",
                "Wheelchair accessibility compliance", 
                "Multiple layout variations",
                "Zone-based furniture grouping",
                "Partner-aware placement"
            ],
            "note": f"This is layout variation {i+1} prioritizing functional relationships between furniture items"
        }
        
        with open(filename, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Layout {i+1} saved to {filename}")

if __name__ == "__main__":
    print("🏠 Creating Multiple Relationship-Aware Deterministic Furniture Layouts...")
    print(f"Room dimensions: {ROOM_WIDTH}cm x {ROOM_HEIGHT}cm")
    print(f"Number of furniture items: {len(furniture_items)}")
    print("🎯 This method focuses on functional relationships between furniture!")
    
    initial_layout = load_initial_layout()
    print("\n🔄 Generating 5 relationship-optimized layout variations...")
    optimized_layouts, planners = create_multiple_relationship_aware_layouts(initial_layout, 5)
    
    # Analyze all layouts
    analyze_layout(initial_layout, planners[0], "Original Layout Analysis")
    
    for i, (layout, planner) in enumerate(zip(optimized_layouts, planners)):
        analyze_layout(layout, planner, f"Relationship-Optimized Layout {i+1} Analysis")
    
    # Compare layouts
    print("\n Layout Comparison Summary:")
    print("="*70)
    print(f"{'Layout':<15} {'Score':<10} {'Accessible':<12} {'Utilization':<12} {'Optimal Rel.':<12}")
    print("-"*70)
    
    original_score = planners[0].evaluate_layout(initial_layout)
    original_accessible = sum(1 for f in initial_layout if planners[0].has_wheelchair_access(f, initial_layout))
    original_utilization = sum(planners[0].get_furniture_dimensions(f)[0] * planners[0].get_furniture_dimensions(f)[1] for f in initial_layout) / (ROOM_WIDTH * ROOM_HEIGHT) * 100
    original_relationships = len(planners[0].analyze_relationships(initial_layout)["optimal_relationships"])
    
    print(f"{'Original':<15} {original_score:<10.1f} {original_accessible:<12} {original_utilization:<12.1f} {original_relationships:<12}")
    
    for i, (layout, planner) in enumerate(zip(optimized_layouts, planners)):
        score = planner.evaluate_layout(layout)
        accessible = sum(1 for f in layout if planner.has_wheelchair_access(f, layout))
        utilization = sum(planner.get_furniture_dimensions(f)[0] * planner.get_furniture_dimensions(f)[1] for f in layout) / (ROOM_WIDTH * ROOM_HEIGHT) * 100
        relationships = len(planner.analyze_relationships(layout)["optimal_relationships"])
        print(f"{'Layout ' + str(i+1):<15} {score:<10.1f} {accessible:<12} {utilization:<12.1f} {relationships:<12}")
    
    save_multiple_layouts(optimized_layouts)
    
    print("\n Generating visualization with all layout variations...")
    visualize_multiple_layouts(optimized_layouts, planners)
    
    print("\n Multiple relationship-aware optimization complete!")
    print(" Run this script again - you'll get 5 different but consistently optimized layouts!")
    print(" Key features:")
    print("   • 5 different optimized layout variations")
    print("   • Functional relationship prioritization")
    print("   • Tables placed with chairs")
    print("   • Beds placed with nightstands") 
    print("   • Desks placed with chairs")
    print("   • Sofas placed with coffee tables")
    print("   • Zone-based furniture grouping")
    print("   • Consistent, repeatable results with variations")
    print("   • Wheelchair accessibility compliance")
    print("   • Optimal distance relationships")
    print("\n Visualization Legend:")
    print("   • Green lines: Optimal relationships")
    print("   • Red dashed lines: Suboptimal relationships") 
    print("   • Green stars: Wheelchair accessible")
    print("   • Red stars: Limited accessibility")
    print("   • Color-coded furniture by functional zones")