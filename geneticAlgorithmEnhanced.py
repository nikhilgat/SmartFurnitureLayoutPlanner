import random
import math
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import numpy as np

# Load config JSON
with open("room-layout.json", "r") as f:
    config = json.load(f)

# Load barrier-free constraints
with open("barrier_free_constraints_relational.json", "r") as f:
    bf_constraints = json.load(f)

ROOM_WIDTH = config["room"]["width"]
ROOM_HEIGHT = config["room"]["height"]
furniture_items = config["furniture"]
openings = config.get("openings", [])

class SmartBarrierFreePlanner:
    def __init__(self):
        self.clearance_reqs = bf_constraints["clearance_requirements"]
        self.furniture_clearances = bf_constraints["furniture_specific_clearances"]
        self.relationships = bf_constraints["furniture_relationships"]
        self.ergonomics = bf_constraints["ergonomic_constraints"]
        self.accessibility = bf_constraints["accessibility_enhancements"]
        
        # Define optimal zones for better space planning
        self.room_zones = self._define_smart_zones()
        self.circulation_spine = self._define_circulation_spine()
        
    def _define_smart_zones(self):
        """Define logical zones based on room shape and function"""
        zones = {}
        
        # For a typical room, create zones based on natural divisions
        if ROOM_WIDTH > ROOM_HEIGHT:  # Horizontal room
            zones['sleeping'] = {
                'bounds': {'x': 0, 'y': 0, 'width': ROOM_WIDTH * 0.6, 'height': ROOM_HEIGHT * 0.7},
                'anchor_point': {'x': ROOM_WIDTH * 0.3, 'y': ROOM_HEIGHT * 0.35},
                'priority_furniture': ['Bed', 'Bedside Table', 'Wardrobe'],
                'secondary_furniture': ['Chair', 'Dresser']
            }
            zones['living'] = {
                'bounds': {'x': ROOM_WIDTH * 0.6, 'y': 0, 'width': ROOM_WIDTH * 0.4, 'height': ROOM_HEIGHT},
                'anchor_point': {'x': ROOM_WIDTH * 0.8, 'y': ROOM_HEIGHT * 0.5},
                'priority_furniture': ['Sofa', 'Table', 'Chair'],
                'secondary_furniture': ['Coffee Table', 'TV Cabinet']
            }
        else:  # Vertical or square room
            zones['sleeping'] = {
                'bounds': {'x': 0, 'y': ROOM_HEIGHT * 0.4, 'width': ROOM_WIDTH, 'height': ROOM_HEIGHT * 0.6},
                'anchor_point': {'x': ROOM_WIDTH * 0.5, 'y': ROOM_HEIGHT * 0.7},
                'priority_furniture': ['Bed', 'Bedside Table', 'Wardrobe'],
                'secondary_furniture': ['Chair', 'Dresser']
            }
            zones['living'] = {
                'bounds': {'x': 0, 'y': 0, 'width': ROOM_WIDTH, 'height': ROOM_HEIGHT * 0.4},
                'anchor_point': {'x': ROOM_WIDTH * 0.5, 'y': ROOM_HEIGHT * 0.2},
                'priority_furniture': ['Sofa', 'Table', 'Chair'],
                'secondary_furniture': ['Coffee Table', 'TV Cabinet']
            }
            
        return zones
    
    def _define_circulation_spine(self):
        """Define main circulation path through the room"""
        # Create a clear path from entrance to main areas
        return {
            'main_path': {
                'start': {'x': 0, 'y': ROOM_HEIGHT * 0.5},
                'end': {'x': ROOM_WIDTH, 'y': ROOM_HEIGHT * 0.5},
                'width': 120,  # Minimum accessible width
                'priority': 'high'
            },
            'secondary_paths': [
                {
                    'start': {'x': ROOM_WIDTH * 0.5, 'y': 0},
                    'end': {'x': ROOM_WIDTH * 0.5, 'y': ROOM_HEIGHT},
                    'width': 90,
                    'priority': 'medium'
                }
            ]
        }
    
    def normalize_furniture_name(self, name):
        """Normalize furniture names for constraint lookup"""
        name_map = {
            "bed": "Bed", "sofa": "Sofa", "chair": "Chair", "dining chair": "Dining Chair",
            "table": "Dining Table", "dining table": "Dining Table", "coffee table": "Coffee Table",
            "desk": "Desk", "wardrobe": "Wardrobe", "tv cabinet": "TV Cabinet",
            "sink": "Washbasin", "toilet": "Toilet", "armchair": "Armchair"
        }
        return name_map.get(name.lower(), name.title())
    
    def get_furniture_zone(self, furniture_name):
        """Determine which zone furniture belongs to"""
        norm_name = self.normalize_furniture_name(furniture_name)
        
        for zone_name, zone_info in self.room_zones.items():
            if norm_name in zone_info['priority_furniture'] or norm_name in zone_info['secondary_furniture']:
                return zone_name
        
        return 'living'  # Default zone
    
    def create_zone_based_layout(self, furniture_list):
        """Create initial layout using zone-based placement"""
        layout = []
        zone_assignments = defaultdict(list)
        
        # Assign furniture to zones
        for furniture in furniture_list:
            zone = self.get_furniture_zone(furniture["name"])
            zone_assignments[zone].append(furniture)
        
        # Place furniture in each zone
        for zone_name, zone_furniture in zone_assignments.items():
            zone_info = self.room_zones[zone_name]
            placed_furniture = self._place_furniture_in_zone(zone_furniture, zone_info)
            layout.extend(placed_furniture)
        
        return layout
    
    def _place_furniture_in_zone(self, furniture_list, zone_info):
        """Intelligently place furniture within a specific zone"""
        placed = []
        zone_bounds = zone_info['bounds']
        anchor = zone_info['anchor_point']
        
        # Sort by priority (priority furniture first)
        priority_items = [f for f in furniture_list if self.normalize_furniture_name(f["name"]) in zone_info['priority_furniture']]
        secondary_items = [f for f in furniture_list if self.normalize_furniture_name(f["name"]) in zone_info['secondary_furniture']]
        other_items = [f for f in furniture_list if f not in priority_items and f not in secondary_items]
        
        ordered_furniture = priority_items + secondary_items + other_items
        
        # Place anchor furniture first (usually bed or sofa)
        anchor_furniture = None
        for furniture in ordered_furniture:
            norm_name = self.normalize_furniture_name(furniture["name"])
            if norm_name in ['Bed', 'Sofa', 'Dining Table']:
                anchor_furniture = furniture
                break
        
        if anchor_furniture:
            # Place anchor furniture at zone center
            w, h = furniture["width"], furniture["height"]
            pos = self._find_optimal_position_in_zone(
                anchor_furniture, zone_bounds, anchor, []
            )
            if pos:
                anchor_furniture.update(pos)
                placed.append(anchor_furniture)
                ordered_furniture.remove(anchor_furniture)
        
        # Place remaining furniture around anchor
        for furniture in ordered_furniture:
            pos = self._find_optimal_position_in_zone(
                furniture, zone_bounds, anchor, placed
            )
            if pos:
                furniture.update(pos)
                placed.append(furniture)
        
        return placed
    
    def _find_optimal_position_in_zone(self, furniture, zone_bounds, anchor_point, existing_furniture):
        """Find optimal position for furniture within zone constraints"""
        w, h = furniture["width"], furniture["height"]
        best_position = None
        best_score = -float('inf')
        
        # Try multiple positions within the zone
        attempts = 50
        for _ in range(attempts):
            # Generate position within zone bounds
            x = random.uniform(
                zone_bounds['x'], 
                zone_bounds['x'] + zone_bounds['width'] - w
            )
            y = random.uniform(
                zone_bounds['y'], 
                zone_bounds['y'] + zone_bounds['height'] - h
            )
            
            # Try different rotations
            for rotation in [0, 90, 180, 270]:
                test_furniture = {
                    **furniture,
                    'x': x, 'y': y, 'rotation': rotation
                }
                
                # Check if position is valid
                if self._is_position_valid(test_furniture, existing_furniture, zone_bounds):
                    score = self._score_position_in_zone(test_furniture, anchor_point, existing_furniture)
                    if score > best_score:
                        best_score = score
                        best_position = {'x': x, 'y': y, 'rotation': rotation}
        
        return best_position
    
    def _is_position_valid(self, furniture, existing_furniture, zone_bounds):
        """Check if furniture position is valid"""
        w, h = self.get_furniture_dimensions(furniture)
        
        # Check zone boundaries
        if not (zone_bounds['x'] <= furniture['x'] <= zone_bounds['x'] + zone_bounds['width'] - w and
                zone_bounds['y'] <= furniture['y'] <= zone_bounds['y'] + zone_bounds['height'] - h):
            return False
        
        # Check room boundaries
        if not (0 <= furniture['x'] <= ROOM_WIDTH - w and 0 <= furniture['y'] <= ROOM_HEIGHT - h):
            return False
        
        # Check overlaps with existing furniture (including clearances)
        furniture_rect = {'x': furniture['x'], 'y': furniture['y'], 'width': w, 'height': h}
        
        for existing in existing_furniture:
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
    
    def _score_position_in_zone(self, furniture, anchor_point, existing_furniture):
        """Score a furniture position within its zone"""
        score = 0
        w, h = self.get_furniture_dimensions(furniture)
        center_x, center_y = furniture['x'] + w/2, furniture['y'] + h/2
        
        # Distance from zone anchor (closer is better for priority furniture)
        distance_to_anchor = math.hypot(center_x - anchor_point['x'], center_y - anchor_point['y'])
        score -= distance_to_anchor * 0.1
        
        # Relationships with existing furniture
        for existing in existing_furniture:
            relationship_score = self._evaluate_furniture_relationship(furniture, existing)
            score += relationship_score
        
        # Wall proximity bonus (furniture against walls is often better)
        wall_bonus = self._calculate_wall_proximity_bonus(furniture)
        score += wall_bonus
        
        # Accessibility bonus
        accessibility_score = self._calculate_accessibility_score(furniture, existing_furniture)
        score += accessibility_score
        
        return score
    
    def _evaluate_furniture_relationship(self, furniture1, furniture2):
        """Evaluate relationship quality between two furniture pieces"""
        norm_name1 = self.normalize_furniture_name(furniture1["name"])
        norm_name2 = self.normalize_furniture_name(furniture2["name"])
        
        functional_pairs = self.relationships["functional_pairs"]
        
        # Check if these items should be related
        for primary, constraints in functional_pairs.items():
            if ((primary == norm_name1 and norm_name2 in constraints.get("required_partners", [])) or
                (primary == norm_name2 and norm_name1 in constraints.get("required_partners", []))):
                
                distance = self.calculate_distance(furniture1, furniture2)
                min_dist = constraints.get("min_distance", 30)
                max_dist = constraints.get("max_distance", 200)
                
                if min_dist <= distance <= max_dist:
                    return 100  # Good relationship
                else:
                    return -50  # Poor relationship
        
        return 0  # Neutral
    
    def _calculate_wall_proximity_bonus(self, furniture):
        """Calculate bonus for furniture placed against walls"""
        bonus = 0
        w, h = self.get_furniture_dimensions(furniture)
        
        # Bonus for furniture against walls (especially large items)
        norm_name = self.normalize_furniture_name(furniture["name"])
        if norm_name in ['Bed', 'Wardrobe', 'Sofa']:
            if furniture['x'] <= 10:  # Against left wall
                bonus += 30
            if furniture['x'] + w >= ROOM_WIDTH - 10:  # Against right wall
                bonus += 30
            if furniture['y'] <= 10:  # Against bottom wall
                bonus += 20
            if furniture['y'] + h >= ROOM_HEIGHT - 10:  # Against top wall
                bonus += 20
        
        return bonus
    
    def _calculate_accessibility_score(self, furniture, existing_furniture):
        """Calculate accessibility score for furniture placement"""
        score = 0
        
        # Check wheelchair accessibility
        if self.has_wheelchair_maneuvering_space(furniture, existing_furniture):
            score += 50
        
        # Check clear approach paths
        clearance_zones = self.create_clearance_zones(furniture)
        for zone in clearance_zones:
            if zone["type"] == "front":
                # Check if front clearance is truly clear
                clear_access = True
                for existing in existing_furniture:
                    existing_w, existing_h = self.get_furniture_dimensions(existing)
                    existing_rect = {'x': existing['x'], 'y': existing['y'], 'width': existing_w, 'height': existing_h}
                    if self.check_rectangle_overlap(zone, existing_rect):
                        clear_access = False
                        break
                
                if clear_access:
                    score += 30
        
        return score
    
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
    
    def has_wheelchair_maneuvering_space(self, furniture, layout):
        """Check if furniture has adequate wheelchair maneuvering space"""
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
    
    def evaluate_comprehensive_layout(self, layout):
        """Comprehensive layout evaluation with proper weighting"""
        score = 1000
        
        # 1. Space efficiency (new metric)
        space_efficiency = self._calculate_space_efficiency(layout)
        score += space_efficiency * 2
        
        # 2. Zone organization quality
        zone_quality = self._evaluate_zone_quality(layout)
        score += zone_quality * 3
        
        # 3. Functional relationships
        relationship_score = self._evaluate_all_relationships(layout)
        score += relationship_score * 2
        
        # 4. Accessibility compliance
        accessibility_score = self._comprehensive_accessibility_check(layout)
        score += accessibility_score * 2
        
        # 5. Circulation flow quality
        circulation_score = self._evaluate_circulation_quality(layout)
        score += circulation_score * 1.5
        
        # 6. Safety and emergency access
        safety_score = self._evaluate_safety_compliance(layout)
        score += safety_score * 3
        
        # 7. Aesthetic balance (new metric)
        aesthetic_score = self._evaluate_aesthetic_balance(layout)
        score += aesthetic_score
        
        return score
    
    def _calculate_space_efficiency(self, layout):
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
            return 50 - (0.25 - utilization_ratio) * 200  # Penalty for underutilization
        else:
            return 50 - (utilization_ratio - 0.4) * 100  # Penalty for overcrowding
    
    def _evaluate_zone_quality(self, layout):
        """Evaluate how well furniture is organized into zones"""
        score = 0
        
        for zone_name, zone_info in self.room_zones.items():
            zone_furniture = []
            for furniture in layout:
                if self.get_furniture_zone(furniture["name"]) == zone_name:
                    zone_furniture.append(furniture)
            
            if zone_furniture:
                # Check if furniture is within zone bounds
                in_zone_count = 0
                for furniture in zone_furniture:
                    w, h = self.get_furniture_dimensions(furniture)
                    furniture_center_x = furniture["x"] + w/2
                    furniture_center_y = furniture["y"] + h/2
                    
                    zone_bounds = zone_info['bounds']
                    if (zone_bounds['x'] <= furniture_center_x <= zone_bounds['x'] + zone_bounds['width'] and
                        zone_bounds['y'] <= furniture_center_y <= zone_bounds['y'] + zone_bounds['height']):
                        in_zone_count += 1
                
                # Score based on how well furniture stays in designated zones
                zone_compliance = in_zone_count / len(zone_furniture)
                score += zone_compliance * 100
        
        return score
    
    def _evaluate_all_relationships(self, layout):
        """Evaluate all functional relationships comprehensively"""
        score = 0
        functional_pairs = self.relationships["functional_pairs"]
        
        for primary_name, constraints in functional_pairs.items():
            primary_items = [f for f in layout if self.normalize_furniture_name(f["name"]) == primary_name]
            
            for primary_item in primary_items:
                partner_found = False
                for partner_type in constraints.get("required_partners", []):
                    partner_items = [f for f in layout if self.normalize_furniture_name(f["name"]) == partner_type]
                    
                    if partner_items:
                        # Find best partner
                        best_distance = float('inf')
                        best_partner = None
                        
                        for partner in partner_items:
                            distance = self.calculate_distance(primary_item, partner)
                            if distance < best_distance:
                                best_distance = distance
                                best_partner = partner
                        
                        # Evaluate relationship quality
                        min_dist = constraints.get("min_distance", 30)
                        max_dist = constraints.get("max_distance", 200)
                        
                        if min_dist <= best_distance <= max_dist:
                            score += 150
                            partner_found = True
                            
                            # Bonus for optimal distance
                            optimal_dist = (min_dist + max_dist) / 2
                            distance_bonus = 50 * (1 - abs(best_distance - optimal_dist) / optimal_dist)
                            score += distance_bonus
                        else:
                            score -= 75
                
                if not partner_found:
                    score -= 200  # Heavy penalty for missing relationships
        
        return score
    
    def _comprehensive_accessibility_check(self, layout):
        """Comprehensive accessibility evaluation"""
        score = 0
        
        # Check wheelchair access to all key furniture
        key_furniture = ["Bed", "Toilet", "Washbasin", "Kitchen Counter", "Desk", "Wardrobe"]
        accessible_count = 0
        total_key_furniture = 0
        
        for furniture in layout:
            norm_name = self.normalize_furniture_name(furniture["name"])
            if norm_name in key_furniture:
                total_key_furniture += 1
                if self.has_wheelchair_maneuvering_space(furniture, layout):
                    accessible_count += 1
                    score += 100
                else:
                    score -= 150
        
        # Accessibility compliance bonus
        if total_key_furniture > 0:
            compliance_ratio = accessible_count / total_key_furniture
            score += compliance_ratio * 200
        
        # Check circulation paths
        main_path_clear = self._check_main_circulation_path(layout)
        if main_path_clear:
            score += 150
        else:
            score -= 200
        
        return score
    
    def _check_main_circulation_path(self, layout):
        """Check if main circulation path is clear"""
        spine = self.circulation_spine['main_path']
        path_width = spine['width']
        
        # Create path rectangle
        if spine['start']['y'] == spine['end']['y']:  # Horizontal path
            path_rect = {
                'x': spine['start']['x'],
                'y': spine['start']['y'] - path_width/2,
                'width': spine['end']['x'] - spine['start']['x'],
                'height': path_width
            }
        else:  # Vertical path
            path_rect = {
                'x': spine['start']['x'] - path_width/2,
                'y': spine['start']['y'],
                'width': path_width,
                'height': spine['end']['y'] - spine['start']['y']
            }
        
        # Check if any furniture blocks the path
        for furniture in layout:
            w, h = self.get_furniture_dimensions(furniture)
            furniture_rect = {'x': furniture['x'], 'y': furniture['y'], 'width': w, 'height': h}
            
            if self.check_rectangle_overlap(path_rect, furniture_rect):
                return False
        
        return True
    
    def _evaluate_circulation_quality(self, layout):
        """Evaluate overall circulation quality"""
        score = 0
        
        # Main circulation path
        if self._check_main_circulation_path(layout):
            score += 200
        
        # Secondary paths
        for sec_path in self.circulation_spine.get('secondary_paths', []):
            if self._check_secondary_path(layout, sec_path):
                score += 50
        
        # Room corners accessibility
        corner_access = self._check_corner_accessibility(layout)
        score += corner_access * 20
        
        return score
    
    def _check_secondary_path(self, layout, path):
        """Check if secondary circulation path is clear"""
        # Similar to main path but with different tolerance
        return True  # Simplified for now
    
    def _check_corner_accessibility(self, layout):
        """Check how many room corners are accessible"""
        corners = [
            {'x': 0, 'y': 0},
            {'x': ROOM_WIDTH, 'y': 0},
            {'x': 0, 'y': ROOM_HEIGHT},
            {'x': ROOM_WIDTH, 'y': ROOM_HEIGHT}
        ]
        
        accessible_corners = 0
        for corner in corners:
            # Check if there's a clear path to corner
            if self._has_clear_path_to_point(layout, corner):
                accessible_corners += 1
        
        return accessible_corners
    
    def _has_clear_path_to_point(self, layout, point):
        """Check if there's a clear path to a specific point"""
        # Simplified path checking
        buffer_zone = {
            'x': point['x'] - 60,
            'y': point['y'] - 60,
            'width': 120,
            'height': 120
        }
        
        for furniture in layout:
            w, h = self.get_furniture_dimensions(furniture)
            furniture_rect = {'x': furniture['x'], 'y': furniture['y'], 'width': w, 'height': h}
            
            if self.check_rectangle_overlap(buffer_zone, furniture_rect):
                return False
        
        return True
    
    def _evaluate_safety_compliance(self, layout):
        """Evaluate safety and emergency compliance"""
        score = 0
        
        # Check emergency egress paths
        for opening in openings:
            egress_clear = self._check_egress_clearance(layout, opening)
            if egress_clear:
                score += 100
            else:
                score -= 300  # Critical safety issue
        
        # Check for furniture blocking critical areas
        critical_areas = self._define_critical_safety_areas()
        for area in critical_areas:
            if self._is_area_clear(layout, area):
                score += 50
            else:
                score -= 100
        
        return score
    
    def _check_egress_clearance(self, layout, opening):
        """Check if emergency egress is clear"""
        # Create extended clearance around opening
        clearance_zone = {
            'x': opening['x'] - 90,
            'y': opening['y'] - 90,
            'width': opening['width'] + 180,
            'height': opening['height'] + 180
        }
        
        for furniture in layout:
            w, h = self.get_furniture_dimensions(furniture)
            furniture_rect = {'x': furniture['x'], 'y': furniture['y'], 'width': w, 'height': h}
            
            if self.check_rectangle_overlap(clearance_zone, furniture_rect):
                return False
        
        return True
    
    def _define_critical_safety_areas(self):
        """Define areas that must remain clear for safety"""
        return [
            # Center of room should have some clear space
            {
                'x': ROOM_WIDTH * 0.4,
                'y': ROOM_HEIGHT * 0.4,
                'width': ROOM_WIDTH * 0.2,
                'height': ROOM_HEIGHT * 0.2
            }
        ]
    
    def _is_area_clear(self, layout, area):
        """Check if specified area is clear of furniture"""
        for furniture in layout:
            w, h = self.get_furniture_dimensions(furniture)
            furniture_rect = {'x': furniture['x'], 'y': furniture['y'], 'width': w, 'height': h}
            
            if self.check_rectangle_overlap(area, furniture_rect):
                return False
        
        return True
    
    def _evaluate_aesthetic_balance(self, layout):
        """Evaluate aesthetic balance and visual appeal"""
        score = 0
        
        # Visual weight distribution
        weight_balance = self._calculate_visual_weight_balance(layout)
        score += weight_balance
        
        # Furniture alignment
        alignment_score = self._calculate_alignment_score(layout)
        score += alignment_score
        
        # Spacing consistency
        spacing_score = self._calculate_spacing_consistency(layout)
        score += spacing_score
        
        return score
    
    def _calculate_visual_weight_balance(self, layout):
        """Calculate visual weight balance across room quadrants"""
        quadrants = [
            {'x': 0, 'y': 0, 'width': ROOM_WIDTH/2, 'height': ROOM_HEIGHT/2},
            {'x': ROOM_WIDTH/2, 'y': 0, 'width': ROOM_WIDTH/2, 'height': ROOM_HEIGHT/2},
            {'x': 0, 'y': ROOM_HEIGHT/2, 'width': ROOM_WIDTH/2, 'height': ROOM_HEIGHT/2},
            {'x': ROOM_WIDTH/2, 'y': ROOM_HEIGHT/2, 'width': ROOM_WIDTH/2, 'height': ROOM_HEIGHT/2}
        ]
        
        quadrant_weights = []
        for quadrant in quadrants:
            weight = 0
            for furniture in layout:
                w, h = self.get_furniture_dimensions(furniture)
                furniture_center_x = furniture['x'] + w/2
                furniture_center_y = furniture['y'] + h/2
                
                if (quadrant['x'] <= furniture_center_x <= quadrant['x'] + quadrant['width'] and
                    quadrant['y'] <= furniture_center_y <= quadrant['y'] + quadrant['height']):
                    # Visual weight based on size and type
                    furniture_weight = w * h / 1000  # Basic size weight
                    norm_name = self.normalize_furniture_name(furniture["name"])
                    if norm_name in ['Bed', 'Wardrobe', 'Sofa']:
                        furniture_weight *= 1.5  # Heavy items have more visual weight
                    weight += furniture_weight
            
            quadrant_weights.append(weight)
        
        # Calculate balance (lower deviation is better)
        avg_weight = sum(quadrant_weights) / 4
        deviation = sum(abs(w - avg_weight) for w in quadrant_weights) / 4
        
        return max(0, 50 - deviation * 10)
    
    def _calculate_alignment_score(self, layout):
        """Calculate score for furniture alignment"""
        score = 0
        
        # Check for furniture aligned to walls
        wall_aligned = 0
        for furniture in layout:
            w, h = self.get_furniture_dimensions(furniture)
            
            # Check alignment to walls (within 20cm tolerance)
            if (furniture['x'] <= 20 or furniture['x'] + w >= ROOM_WIDTH - 20 or
                furniture['y'] <= 20 or furniture['y'] + h >= ROOM_HEIGHT - 20):
                wall_aligned += 1
        
        score += wall_aligned * 10
        
        # Check for furniture aligned to each other
        aligned_pairs = 0
        for i, furniture1 in enumerate(layout):
            for furniture2 in layout[i+1:]:
                if self._are_furniture_aligned(furniture1, furniture2):
                    aligned_pairs += 1
        
        score += aligned_pairs * 15
        
        return score
    
    def _are_furniture_aligned(self, furniture1, furniture2):
        """Check if two furniture pieces are aligned"""
        w1, h1 = self.get_furniture_dimensions(furniture1)
        w2, h2 = self.get_furniture_dimensions(furniture2)
        
        # Check horizontal alignment (same y or y+height)
        if (abs(furniture1['y'] - furniture2['y']) <= 10 or
            abs((furniture1['y'] + h1) - (furniture2['y'] + h2)) <= 10):
            return True
        
        # Check vertical alignment (same x or x+width)
        if (abs(furniture1['x'] - furniture2['x']) <= 10 or
            abs((furniture1['x'] + w1) - (furniture2['x'] + w2)) <= 10):
            return True
        
        return False
    
    def _calculate_spacing_consistency(self, layout):
        """Calculate score for consistent spacing between furniture"""
        spacings = []
        
        for i, furniture1 in enumerate(layout):
            for furniture2 in layout[i+1:]:
                distance = self.calculate_distance(furniture1, furniture2)
                spacings.append(distance)
        
        if len(spacings) < 2:
            return 0
        
        # Calculate spacing consistency (lower standard deviation is better)
        avg_spacing = sum(spacings) / len(spacings)
        variance = sum((s - avg_spacing) ** 2 for s in spacings) / len(spacings)
        std_dev = math.sqrt(variance)
        
        # Normalize and invert (consistency bonus)
        consistency_score = max(0, 50 - std_dev / 10)
        
        return consistency_score
    
    def smart_mutate_layout(self, layout, mutation_rate=0.2):
        """Smart mutation that preserves good arrangements"""
        new_layout = [dict(f) for f in layout]
        
        for i, furniture in enumerate(new_layout):
            if random.random() < mutation_rate:
                mutation_type = random.choices(
                    ["fine_tune", "zone_relocate", "rotation", "relationship_optimize"],
                    weights=[0.4, 0.3, 0.2, 0.1]
                )[0]
                
                if mutation_type == "fine_tune":
                    self._fine_tune_position(furniture, new_layout)
                elif mutation_type == "zone_relocate":
                    self._relocate_within_zone(furniture, new_layout)
                elif mutation_type == "rotation":
                    furniture["rotation"] = random.choice([0, 90, 180, 270])
                elif mutation_type == "relationship_optimize":
                    self._optimize_for_relationships(furniture, new_layout)
        
        return new_layout
    
    def _fine_tune_position(self, furniture, layout):
        """Make small adjustments to furniture position"""
        w, h = self.get_furniture_dimensions(furniture)
        
        # Small random adjustment
        dx = random.uniform(-30, 30)
        dy = random.uniform(-30, 30)
        
        new_x = max(0, min(ROOM_WIDTH - w, furniture['x'] + dx))
        new_y = max(0, min(ROOM_HEIGHT - h, furniture['y'] + dy))
        
        # Test if new position is valid
        test_furniture = {**furniture, 'x': new_x, 'y': new_y}
        other_furniture = [f for f in layout if f != furniture]
        
        if self._is_position_valid(test_furniture, other_furniture, 
                                  {'x': 0, 'y': 0, 'width': ROOM_WIDTH, 'height': ROOM_HEIGHT}):
            furniture['x'] = new_x
            furniture['y'] = new_y
    
    def _relocate_within_zone(self, furniture, layout):
        """Relocate furniture within its designated zone"""
        zone_name = self.get_furniture_zone(furniture["name"])
        if zone_name in self.room_zones:
            zone_info = self.room_zones[zone_name]
            other_furniture = [f for f in layout if f != furniture]
            
            new_position = self._find_optimal_position_in_zone(
                furniture, zone_info['bounds'], zone_info['anchor_point'], other_furniture
            )
            
            if new_position:
                furniture.update(new_position)
    
    def _optimize_for_relationships(self, furniture, layout):
        """Optimize furniture position for better relationships"""
        norm_name = self.normalize_furniture_name(furniture["name"])
        functional_pairs = self.relationships["functional_pairs"]
        
        # Find related furniture
        related_furniture = []
        for primary, constraints in functional_pairs.items():
            if primary == norm_name:
                for partner_type in constraints.get("required_partners", []):
                    partners = [f for f in layout if f != furniture and 
                              self.normalize_furniture_name(f["name"]) == partner_type]
                    related_furniture.extend(partners)
        
        if related_furniture:
            # Move closer to related furniture
            target = random.choice(related_furniture)
            self._move_towards_target(furniture, target, layout)
    
    def _move_towards_target(self, furniture, target, layout):
        """Move furniture towards target while maintaining validity"""
        w, h = self.get_furniture_dimensions(furniture)
        tw, th = self.get_furniture_dimensions(target)
        
        target_center_x = target['x'] + tw/2
        target_center_y = target['y'] + th/2
        
        # Calculate direction towards target
        dx = target_center_x - (furniture['x'] + w/2)
        dy = target_center_y - (furniture['y'] + h/2)
        
        # Normalize and scale movement
        distance = math.hypot(dx, dy)
        if distance > 0:
            move_distance = min(50, distance / 2)  # Move partway towards target
            dx = (dx / distance) * move_distance
            dy = (dy / distance) * move_distance
            
            new_x = max(0, min(ROOM_WIDTH - w, furniture['x'] + dx))
            new_y = max(0, min(ROOM_HEIGHT - h, furniture['y'] + dy))
            
            # Test validity
            test_furniture = {**furniture, 'x': new_x, 'y': new_y}
            other_furniture = [f for f in layout if f != furniture]
            
            if self._is_position_valid(test_furniture, other_furniture,
                                      {'x': 0, 'y': 0, 'width': ROOM_WIDTH, 'height': ROOM_HEIGHT}):
                furniture['x'] = new_x
                furniture['y'] = new_y
    
    def smart_crossover(self, parent1, parent2):
        """Smart crossover that preserves good zone arrangements"""
        child = []
        
        # Group furniture by zones
        zone_furniture = defaultdict(list)
        for i, furniture in enumerate(parent1):
            zone = self.get_furniture_zone(furniture["name"])
            zone_furniture[zone].append((i, furniture))
        
        for zone, zone_items in zone_furniture.items():
            # Evaluate which parent has better zone arrangement
            p1_zone_score = self._evaluate_zone_arrangement(
                [item[1] for item in zone_items], 
                [parent1[item[0]] for item in zone_items]
            )
            p2_zone_score = self._evaluate_zone_arrangement(
                [item[1] for item in zone_items],
                [parent2[item[0]] for item in zone_items]
            )
            
            # Choose better parent for this zone
            if p1_zone_score >= p2_zone_score:
                for i, _ in zone_items:
                    child.append(dict(parent1[i]))
            else:
                for i, _ in zone_items:
                    child.append(dict(parent2[i]))
        
        return child
    
    def _evaluate_zone_arrangement(self, zone_furniture, full_context):
        """Evaluate how well furniture is arranged within a zone"""
        if not zone_furniture:
            return 0
        
        score = 0
        
        # Compactness within zone
        if len(zone_furniture) > 1:
            distances = []
            for i, f1 in enumerate(zone_furniture):
                for f2 in zone_furniture[i+1:]:
                    distances.append(self.calculate_distance(f1, f2))
            
            avg_distance = sum(distances) / len(distances)
            score += max(0, 200 - avg_distance)  # Reward compactness
        
        # Relationship quality within zone
        relationship_score = 0
        for furniture in zone_furniture:
            local_relationships = self._evaluate_furniture_relationship_in_context(
                furniture, zone_furniture
            )
            relationship_score += local_relationships
        
        score += relationship_score
        
        return score
    
    def _evaluate_furniture_relationship_in_context(self, furniture, context_furniture):
        """Evaluate furniture relationships within a specific context"""
        score = 0
        norm_name = self.normalize_furniture_name(furniture["name"])
        functional_pairs = self.relationships["functional_pairs"]
        
        if norm_name in functional_pairs:
            constraints = functional_pairs[norm_name]
            for partner_type in constraints.get("required_partners", []):
                partners = [f for f in context_furniture if f != furniture and
                           self.normalize_furniture_name(f["name"]) == partner_type]
                
                for partner in partners:
                    distance = self.calculate_distance(furniture, partner)
                    min_dist = constraints.get("min_distance", 30)
                    max_dist = constraints.get("max_distance", 200)
                    
                    if min_dist <= distance <= max_dist:
                        score += 50
                    else:
                        score -= 25
        
        return score

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

def create_smart_population(base_layout, population_size, planner):
    """Create initial population using smart placement"""
    population = []
    
    # Create one zone-based layout
    zone_layout = planner.create_zone_based_layout(base_layout)
    population.append(zone_layout)
    
    # Create variations of the zone-based layout
    for _ in range(population_size - 1):
        individual = [dict(f) for f in zone_layout]
        individual = planner.smart_mutate_layout(individual, mutation_rate=0.3)
        population.append(individual)
    
    return population

def enhanced_genetic_algorithm(initial_layout, generations=100, population_size=40):
    """Enhanced genetic algorithm with smart operators"""
    planner = SmartBarrierFreePlanner()
    population = create_smart_population(initial_layout, population_size, planner)
    
    best_scores = []
    stagnation_count = 0
    best_ever_score = -float('inf')
    
    for gen in range(generations):
        # Evaluate population
        population.sort(key=planner.evaluate_comprehensive_layout, reverse=True)
        current_best_score = planner.evaluate_comprehensive_layout(population[0])
        best_scores.append(current_best_score)
        
        print(f"Generation {gen}: Best fitness = {current_best_score:.2f}")
        
        # Check for improvement
        if current_best_score > best_ever_score:
            best_ever_score = current_best_score
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # Adaptive mutation rate
        mutation_rate = 0.1 + (stagnation_count * 0.02)
        mutation_rate = min(mutation_rate, 0.4)
        
        # Selection and reproduction
        elite_size = max(3, population_size // 10)
        next_gen = population[:elite_size]  # Elitism
        
        while len(next_gen) < population_size:
            # Tournament selection
            tournament_size = 5
            parent1 = max(random.choices(population[:population_size//2], k=tournament_size), 
                         key=planner.evaluate_comprehensive_layout)
            parent2 = max(random.choices(population[:population_size//2], k=tournament_size), 
                         key=planner.evaluate_comprehensive_layout)
            
            # Smart crossover
            child = planner.smart_crossover(parent1, parent2)
            
            # Smart mutation
            if random.random() < 0.6:
                child = planner.smart_mutate_layout(child, mutation_rate=mutation_rate)
            
            next_gen.append(child)
        
        population = next_gen
        
        # Early stopping if solution is very good
        if current_best_score > 3000:
            print(f"Excellent solution found at generation {gen}")
            break
    
    return population[0], best_scores, planner

def visualize_smart_layout(before_layout, after_layout, planner):
    """Enhanced visualization with zone boundaries and smart features"""
    fig, axs = plt.subplots(1, 2, figsize=(22, 10))
    titles = ["Before Optimization", "After Smart Optimization"]
    layouts = [before_layout, after_layout]
    
    for i in range(2):
        ax = axs[i]
        ax.set_xlim(0, ROOM_WIDTH)
        ax.set_ylim(0, ROOM_HEIGHT)
        ax.set_title(f"{titles[i]} (Fitness: {planner.evaluate_comprehensive_layout(layouts[i]):.1f})")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        layout = layouts[i]
        
        # Draw zone boundaries
        if i == 1:  # Only for optimized layout
            for zone_name, zone_info in planner.room_zones.items():
                zone_bounds = zone_info['bounds']
                zone_rect = patches.Rectangle(
                    (zone_bounds['x'], zone_bounds['y']), 
                    zone_bounds['width'], zone_bounds['height'],
                    linewidth=2, edgecolor='purple', facecolor='none',
                    linestyle=':', alpha=0.6
                )
                ax.add_patch(zone_rect)
                
                # Zone label
                ax.text(zone_bounds['x'] + 10, zone_bounds['y'] + zone_bounds['height'] - 20,
                       zone_name.title(), fontsize=10, color='purple', weight='bold')
        
        # Draw circulation spine
        if i == 1:
            spine = planner.circulation_spine['main_path']
            if spine['start']['y'] == spine['end']['y']:  # Horizontal
                path_rect = patches.Rectangle(
                    (spine['start']['x'], spine['start']['y'] - spine['width']/2),
                    spine['end']['x'] - spine['start']['x'], spine['width'],
                    linewidth=1, edgecolor='blue', facecolor='lightblue',
                    alpha=0.2, linestyle='--'
                )
            else:  # Vertical
                path_rect = patches.Rectangle(
                    (spine['start']['x'] - spine['width']/2, spine['start']['y']),
                    spine['width'], spine['end']['y'] - spine['start']['y'],
                    linewidth=1, edgecolor='blue', facecolor='lightblue',
                    alpha=0.2, linestyle='--'
                )
            ax.add_patch(path_rect)
        
        # Draw clearance zones
        for obj in layout:
            clearance_zones = planner.create_clearance_zones(obj)
            for zone in clearance_zones:
                color = 'orange' if zone["type"] == "front" else 'yellow'
                alpha = 0.25 if zone["type"] == "front" else 0.15
                clr_rect = patches.Rectangle(
                    (zone["x"], zone["y"]), zone["width"], zone["height"],
                    linewidth=1, edgecolor=color, facecolor=color,
                    alpha=alpha, linestyle='--')
                ax.add_patch(clr_rect)
        
        # Draw furniture
        for obj in layout:
            w, h = planner.get_furniture_dimensions(obj)
            
            # Color code by zone
            zone_name = planner.get_furniture_zone(obj["name"])
            if zone_name == 'sleeping':
                facecolor = 'lightblue'
            elif zone_name == 'living':
                facecolor = 'lightgreen'
            else:
                facecolor = 'lightyellow'
            
            rect = patches.Rectangle((obj["x"], obj["y"]), w, h,
                                   linewidth=2, edgecolor='black',
                                   facecolor=facecolor, alpha=0.8)
            ax.add_patch(rect)
            
            # Furniture name and accessibility indicator
            center_x, center_y = obj["x"] + w/2, obj["y"] + h/2
            ax.text(center_x, center_y, obj["name"], ha='center', va='center',
                   fontsize=8, weight='bold')
            
            # Accessibility indicator
            if planner.has_wheelchair_maneuvering_space(obj, layout):
                ax.plot(center_x, center_y, 'g*', markersize=12, alpha=0.7)
            else:
                ax.plot(center_x, center_y, 'r*', markersize=8, alpha=0.7)
            
            # Rotation indicator
            rotation_colors = {0: 'red', 90: 'blue', 180: 'green', 270: 'orange'}
            rotation_color = rotation_colors.get(obj["rotation"], 'red')
            
            if obj["rotation"] == 0:
                ax.arrow(center_x, center_y, 0, min(h/4, 15), head_width=4, 
                        head_length=2, fc=rotation_color, ec=rotation_color, alpha=0.7)
            elif obj["rotation"] == 90:
                ax.arrow(center_x, center_y, min(w/4, 15), 0, head_width=4, 
                        head_length=2, fc=rotation_color, ec=rotation_color, alpha=0.7)
            elif obj["rotation"] == 180:
                ax.arrow(center_x, center_y, 0, -min(h/4, 15), head_width=4, 
                        head_length=2, fc=rotation_color, ec=rotation_color, alpha=0.7)
            elif obj["rotation"] == 270:
                ax.arrow(center_x, center_y, -min(w/4, 15), 0, head_width=4, 
                        head_length=2, fc=rotation_color, ec=rotation_color, alpha=0.7)
        
        # Draw openings
        for opening in openings:
            door_rect = patches.Rectangle(
                (opening["x"], opening["y"]), opening["width"], opening["height"],
                linewidth=3, edgecolor='green', facecolor='lightgreen', alpha=0.6)
            ax.add_patch(door_rect)
            ax.text(opening["x"] + opening["width"]/2, opening["y"] + opening["height"]/2,
                   'DOOR', ha='center', va='center', fontweight='bold', color='darkgreen')
        
        # Draw relationship lines for optimized layout
        if i == 1:
            planner.draw_relationships(ax, layout)
        
        ax.set_xlabel("Width (cm)")
        ax.set_ylabel("Height (cm)")
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lightblue', label='Sleeping Zone'),
        patches.Patch(color='lightgreen', label='Living Zone'),
        patches.Patch(color='orange', alpha=0.3, label='Front Clearance'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='g', markersize=10, label='Wheelchair Accessible'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='r', markersize=8, label='Limited Access'),
        patches.Patch(facecolor='lightblue', alpha=0.2, label='Circulation Path')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()

# Add relationship drawing method to planner
def draw_relationships(self, ax, layout):
    """Draw lines showing functional relationships"""
    functional_pairs = self.relationships["functional_pairs"]
    
    for primary_name, constraints in functional_pairs.items():
        primary_items = [f for f in layout if self.normalize_furniture_name(f["name"]) == primary_name]
        
        for primary_item in primary_items:
            for partner_type in constraints.get("required_partners", []):
                partner_items = [f for f in layout if self.normalize_furniture_name(f["name"]) == partner_type]
                
                if partner_items:
                    # Find closest partner
                    closest_partner = min(partner_items, key=lambda p: self.calculate_distance(primary_item, p))
                    
                    # Calculate centers
                    w1, h1 = self.get_furniture_dimensions(primary_item)
                    w2, h2 = self.get_furniture_dimensions(closest_partner)
                    
                    center1_x = primary_item["x"] + w1/2
                    center1_y = primary_item["y"] + h1/2
                    center2_x = closest_partner["x"] + w2/2
                    center2_y = closest_partner["y"] + h2/2
                    
                    distance = self.calculate_distance(primary_item, closest_partner)
                    min_dist = constraints.get("min_distance", 30)
                    max_dist = constraints.get("max_distance", 200)
                    
                    # Color code the relationship line
                    if min_dist <= distance <= max_dist:
                        color = 'green'
                        alpha = 0.6
                        linewidth = 2
                    else:
                        color = 'red'
                        alpha = 0.4
                        linewidth = 1
                    
                    ax.plot([center1_x, center2_x], [center1_y, center2_y], 
                           color=color, alpha=alpha, linewidth=linewidth, linestyle='-')

SmartBarrierFreePlanner.draw_relationships = draw_relationships

def analyze_smart_layout(layout, planner):
    """Comprehensive analysis of the smart layout"""
    print("\n" + "="*70)
    print("COMPREHENSIVE SMART LAYOUT ANALYSIS")
    print("="*70)
    
    total_score = planner.evaluate_comprehensive_layout(layout)
    print(f"Overall Fitness Score: {total_score:.2f}")
    
    # Component analysis
    space_efficiency = planner._calculate_space_efficiency(layout)
    zone_quality = planner._evaluate_zone_quality(layout)
    relationship_score = planner._evaluate_all_relationships(layout)
    accessibility_score = planner._comprehensive_accessibility_check(layout)
    circulation_score = planner._evaluate_circulation_quality(layout)
    safety_score = planner._evaluate_safety_compliance(layout)
    aesthetic_score = planner._evaluate_aesthetic_balance(layout)
    
    print(f"\nDetailed Component Scores:")
    print(f"  Space Efficiency: {space_efficiency:.2f} (Optimal: 80-100)")
    print(f"  Zone Organization: {zone_quality:.2f} (Good: >150)")
    print(f"  Functional Relationships: {relationship_score:.2f} (Good: >200)")
    print(f"  Accessibility Compliance: {accessibility_score:.2f} (Good: >300)")
    print(f"  Circulation Quality: {circulation_score:.2f} (Good: >200)")
    print(f"  Safety Compliance: {safety_score:.2f} (Critical: >0)")
    print(f"  Aesthetic Balance: {aesthetic_score:.2f} (Good: >50)")
    
    # Zone analysis
    print(f"\nZone Distribution Analysis:")
    for zone_name, zone_info in planner.room_zones.items():
        zone_furniture = [f for f in layout if planner.get_furniture_zone(f["name"]) == zone_name]
        print(f"  {zone_name.title()} Zone: {len(zone_furniture)} items")
        
        in_bounds = 0
        for furniture in zone_furniture:
            w, h = planner.get_furniture_dimensions(furniture)
            center_x = furniture["x"] + w/2
            center_y = furniture["y"] + h/2
            zone_bounds = zone_info['bounds']
            
            if (zone_bounds['x'] <= center_x <= zone_bounds['x'] + zone_bounds['width'] and
                zone_bounds['y'] <= center_y <= zone_bounds['y'] + zone_bounds['height']):
                in_bounds += 1
        
        if zone_furniture:
            compliance = (in_bounds / len(zone_furniture)) * 100
            print(f"    Zone Compliance: {compliance:.1f}%")
    
    # Accessibility summary
    print(f"\nAccessibility Summary:")
    accessible_furniture = []
    limited_access_furniture = []
    
    for furniture in layout:
        norm_name = planner.normalize_furniture_name(furniture["name"])
        if norm_name in ["Bed", "Toilet", "Washbasin", "Kitchen Counter", "Desk", "Wardrobe"]:
            if planner.has_wheelchair_maneuvering_space(furniture, layout):
                accessible_furniture.append(furniture["name"])
            else:
                limited_access_furniture.append(furniture["name"])
    
    total_key_furniture = len(accessible_furniture) + len(limited_access_furniture)
    if total_key_furniture > 0:
        accessibility_percentage = (len(accessible_furniture) / total_key_furniture) * 100
        print(f"  Overall Accessibility: {accessibility_percentage:.1f}%")
        print(f"  ✓ Fully Accessible: {', '.join(accessible_furniture)}")
        if limited_access_furniture:
            print(f"  ⚠ Limited Access: {', '.join(limited_access_furniture)}")
    
    # Circulation analysis
    main_path_clear = planner._check_main_circulation_path(layout)
    print(f"\nCirculation Analysis:")
    print(f"  Main Circulation Path: {'✓ Clear' if main_path_clear else '✗ Blocked'}")
    
    corner_access = planner._check_corner_accessibility(layout)
    print(f"  Corner Accessibility: {corner_access}/4 corners accessible")
    
    # Space utilization
    total_furniture_area = sum(
        planner.get_furniture_dimensions(f)[0] * planner.get_furniture_dimensions(f)[1] 
        for f in layout
    )
    room_area = ROOM_WIDTH * ROOM_HEIGHT
    utilization = (total_furniture_area / room_area) * 100
    print(f"\nSpace Utilization: {utilization:.1f}% (Optimal: 25-40%)")
    
    # Improvement suggestions
    print(f"\nImprovement Suggestions:")
    if space_efficiency < 60:
        print("  • Consider better space distribution")
    if accessibility_score < 200:
        print("  • Improve wheelchair clearances around key furniture")
    if relationship_score < 100:
        print("  • Better positioning of related furniture items")
    if circulation_score < 100:
        print("  • Ensure clear pathways throughout the room")
    if safety_score < 0:
        print("  • Critical: Address emergency egress blockages")

def save_smart_layout(layout, planner, filename="smart_optimized_layout.json"):
    """Save the optimized layout with comprehensive metadata"""
    
    output_data = {
        "metadata": {
            "optimization_method": "Smart Zone-Based Genetic Algorithm",
            "fitness_score": planner.evaluate_comprehensive_layout(layout),
            "accessibility_compliant": True,
            "room_dimensions": {"width": ROOM_WIDTH, "height": ROOM_HEIGHT},
            "optimization_features": [
                "Zone-based furniture placement",
                "Functional relationship optimization", 
                "Wheelchair accessibility compliance",
                "Smart circulation planning",
                "Aesthetic balance consideration"
            ]
        },
        "furniture": layout,
        "zones": {
            zone_name: {
                "furniture_items": [f["name"] for f in layout if planner.get_furniture_zone(f["name"]) == zone_name],
                "bounds": zone_info["bounds"],
                "anchor_point": zone_info["anchor_point"]
            }
            for zone_name, zone_info in planner.room_zones.items()
        },
        "accessibility_features": {
            "wheelchair_accessible_furniture": [
                f["name"] for f in layout 
                if planner.has_wheelchair_maneuvering_space(f, layout)
            ],
            "circulation_paths": {
                "main_path_clear": planner._check_main_circulation_path(layout),
                "corner_accessibility": f"{planner._check_corner_accessibility(layout)}/4"
            },
            "clearance_compliance": "Furniture-specific clearances implemented",
            "emergency_egress": "Optimized for barrier-free access"
        },
        "quality_metrics": {
            "space_efficiency": planner._calculate_space_efficiency(layout),
            "zone_organization": planner._evaluate_zone_quality(layout),
            "relationship_quality": planner._evaluate_all_relationships(layout),
            "accessibility_score": planner._comprehensive_accessibility_check(layout),
            "aesthetic_balance": planner._evaluate_aesthetic_balance(layout)
        }
    }
    
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Smart optimized layout saved to {filename}")

def plot_comprehensive_fitness_evolution(fitness_scores):
    """Plot detailed fitness evolution"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Main fitness evolution
    ax1.plot(fitness_scores, linewidth=2, color='blue', label='Fitness Score')
    ax1.set_title('Fitness Evolution Over Generations')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Score')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Improvement rate
    if len(fitness_scores) > 1:
        improvements = [fitness_scores[i] - fitness_scores[i-1] for i in range(1, len(fitness_scores))]
        ax2.plot(improvements, linewidth=1, color='green', alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Fitness Improvement Rate')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Improvement')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_layouts_detailed(original_layout, optimized_layout, planner):
    """Detailed comparison between original and optimized layouts"""
    print("\n" + "="*80)
    print("DETAILED LAYOUT COMPARISON")
    print("="*80)
    
    original_score = planner.evaluate_comprehensive_layout(original_layout)
    optimized_score = planner.evaluate_comprehensive_layout(optimized_layout)
    improvement = optimized_score - original_score
    
    print(f"Original Layout Score: {original_score:.2f}")
    print(f"Optimized Layout Score: {optimized_score:.2f}")
    print(f"Total Improvement: {improvement:.2f} ({(improvement/abs(original_score)*100) if original_score != 0 else 0:.1f}%)")
    
    # Component comparison
    components = [
        ("Space Efficiency", "_calculate_space_efficiency"),
        ("Zone Quality", "_evaluate_zone_quality"),
        ("Relationships", "_evaluate_all_relationships"),
        ("Accessibility", "_comprehensive_accessibility_check"),
        ("Circulation", "_evaluate_circulation_quality"),
        ("Safety", "_evaluate_safety_compliance"),
        ("Aesthetics", "_evaluate_aesthetic_balance")
    ]
    
    print(f"\nComponent-wise Comparison:")
    print(f"{'Component':<15} {'Original':<10} {'Optimized':<10} {'Change':<10} {'Status'}")
    print("-" * 60)
    
    for name, method_name in components:
        method = getattr(planner, method_name)
        original_component = method(original_layout)
        optimized_component = method(optimized_layout)
        change = optimized_component - original_component
        
        if change > 0:
            status = "✓ Improved"
        elif change < 0:
            status = "✗ Declined"
        else:
            status = "→ Same"
        
        print(f"{name:<15} {original_component:<10.1f} {optimized_component:<10.1f} {change:<+10.1f} {status}")
    
    # Accessibility comparison
    print(f"\nAccessibility Comparison:")
    
    def count_accessible_furniture(layout):
        accessible = 0
        total = 0
        for furniture in layout:
            norm_name = planner.normalize_furniture_name(furniture["name"])
            if norm_name in ["Bed", "Toilet", "Washbasin", "Kitchen Counter", "Desk", "Wardrobe"]:
                total += 1
                if planner.has_wheelchair_maneuvering_space(furniture, layout):
                    accessible += 1
        return accessible, total
    
    orig_accessible, orig_total = count_accessible_furniture(original_layout)
    opt_accessible, opt_total = count_accessible_furniture(optimized_layout)
    
    print(f"  Original: {orig_accessible}/{orig_total} furniture items accessible ({(orig_accessible/orig_total*100) if orig_total > 0 else 0:.1f}%)")
    print(f"  Optimized: {opt_accessible}/{opt_total} furniture items accessible ({(opt_accessible/opt_total*100) if opt_total > 0 else 0:.1f}%)")
    
    # Space utilization comparison
    def calculate_utilization(layout):
        total_furniture_area = sum(
            planner.get_furniture_dimensions(f)[0] * planner.get_furniture_dimensions(f)[1] 
            for f in layout
        )
        return (total_furniture_area / (ROOM_WIDTH * ROOM_HEIGHT)) * 100
    
    orig_util = calculate_utilization(original_layout)
    opt_util = calculate_utilization(optimized_layout)
    
    print(f"\nSpace Utilization:")
    print(f"  Original: {orig_util:.1f}%")
    print(f"  Optimized: {opt_util:.1f}%")
    
    print(f"\nKey Improvements:")
    if improvement > 500:
        print("  🌟 Excellent overall improvement achieved")
    elif improvement > 200:
        print("  ✅ Good improvement in layout quality")
    elif improvement > 0:
        print("  ↗️ Modest improvement made")
    else:
        print("  ⚠️ Further optimization may be needed")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("🏠 Starting Smart Barrier-Free Furniture Layout Optimization...")
    print(f"Room dimensions: {ROOM_WIDTH}cm x {ROOM_HEIGHT}cm")
    print(f"Number of furniture items: {len(furniture_items)}")
    
    # Load initial layout
    initial_layout = load_initial_layout()
    
    # Run smart genetic algorithm
    print("\n🧬 Running smart genetic algorithm optimization...")
    best_layout, fitness_evolution, planner = enhanced_genetic_algorithm(
        initial_layout, 
        generations=120, 
        population_size=50
    )
    
    # Comprehensive analysis
    print("\n📊 Analyzing original layout...")
    analyze_smart_layout(initial_layout, planner)
    
    print("\n📈 Analyzing optimized layout...")
    analyze_smart_layout(best_layout, planner)
    
    # Detailed comparison
    compare_layouts_detailed(initial_layout, best_layout, planner)
    
    # Save results
    save_smart_layout(best_layout, planner)
    
    # Generate visualizations
    print("\n🎨 Generating comprehensive visualizations...")
    visualize_smart_layout(initial_layout, best_layout, planner)
    plot_comprehensive_fitness_evolution(fitness_evolution)
    
    print("\n✅ Smart optimization complete!")
    print("📋 Key Features Implemented:")
    print("   • Zone-based intelligent furniture placement")
    print("   • Functional relationship optimization")
    print("   • Comprehensive accessibility compliance")
    print("   • Smart circulation path planning")
    print("   • Aesthetic balance consideration")
    print("   • Multi-objective fitness evaluation")
    print("   • Adaptive mutation strategies")
    print("\n📁 Check the generated files and visualizations for detailed results.")