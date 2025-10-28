"""
Scientific Cost Function Validator for Facility Layout Optimization
Based on Multi-Objective Unequal-Area Facility Layout Problem (UA-FLP)

Implements the total cost function:
C_total(L) = w1*C_flow + w2*C_zone + w3*C_env + w4*C_clearance + w5*C_vis

Usage: python cost_validator.py <original_layout.json> <optimized_layout.json>
"""

import json
import sys
import math
from shapely.geometry import box, Point, Polygon
from shapely.affinity import rotate as shapely_rotate


class CostFunctionValidator:
    """
    Validates and compares layouts using multi-objective cost function
    """
    
    # ========================================================================
    # CONFIGURATION: Weights for cost components (should sum to 1.0)
    # ========================================================================
    WEIGHTS = {
        'w1': 0.25,  # Flow and adjacency
        'w2': 0.20,  # Zoning
        'w3': 0.20,  # Environmental
        'w4': 0.25,  # Clearance/Ergonomics
        'w5': 0.10   # Aesthetics
    }
    
    # Affinity matrix: importance of co-location (0-100)
    AFFINITY_MATRIX = {
        ('bed', 'bedside'): 100,
        ('study table', 'study chair'): 100,
        ('sofa', 'sofa'): 60,
        ('bed', 'wardrobe'): 40,
        ('study table', 'window'): 70,
        # Add more relationships as needed
    }
    
    # Penalty constants
    PENALTY_WINDOW_OBSTRUCTION = 1000
    PENALTY_DOOR_SWING_BLOCK = 100000
    PENALTY_CHAIR_PULLOUT_BLOCK = 50000
    PENALTY_VIEW_BLOCK = 500
    PENALTY_CENTRAL_PLACEMENT = 300
    
    # Ergonomic standards (cm)
    DOOR_SWING_CLEARANCE = 90
    CHAIR_PULLOUT_DEPTH = 80
    
    def __init__(self, layout_data):
        self.room = layout_data['room']
        self.furniture = layout_data['furniture']
        self.openings = layout_data['openings']
        
        # Calculate zone centroids based on room dimensions
        self._calculate_zones()
    
    def _calculate_zones(self):
        """
        Define functional zones based on room geometry and door position.
        Zones: Private Zone 1, Private Zone 2, Shared Zone
        """
        w, h = self.room['width'], self.room['height']
        
        # Find door position
        door = next((o for o in self.openings if o['type'] == 'door'), None)
        
        if door:
            door_wall = door['wall']
        else:
            door_wall = 'bottom'  # Default assumption
        
        # Define zone centroids (μ) based on door position
        # Zone 1: Sleeping area (opposite from door)
        # Zone 2: Activity area (near window)
        # Zone S: Shared area (central)
        
        if door_wall == 'right':
            self.zones = {
                'private1': {'centroid': (w * 0.25, h * 0.25), 'furniture': ['bed', 'bedside']},
                'private2': {'centroid': (w * 0.25, h * 0.75), 'furniture': ['study table', 'study chair']},
                'shared': {'centroid': (w * 0.6, h * 0.5), 'furniture': ['sofa']}
            }
        elif door_wall == 'left':
            self.zones = {
                'private1': {'centroid': (w * 0.75, h * 0.25), 'furniture': ['bed', 'bedside']},
                'private2': {'centroid': (w * 0.75, h * 0.75), 'furniture': ['study table', 'study chair']},
                'shared': {'centroid': (w * 0.4, h * 0.5), 'furniture': ['sofa']}
            }
        elif door_wall == 'top':
            self.zones = {
                'private1': {'centroid': (w * 0.25, h * 0.75), 'furniture': ['bed', 'bedside']},
                'private2': {'centroid': (w * 0.75, h * 0.75), 'furniture': ['study table', 'study chair']},
                'shared': {'centroid': (w * 0.5, h * 0.4), 'furniture': ['sofa']}
            }
        else:  # bottom
            self.zones = {
                'private1': {'centroid': (w * 0.25, h * 0.25), 'furniture': ['bed', 'bedside']},
                'private2': {'centroid': (w * 0.75, h * 0.25), 'furniture': ['study table', 'study chair']},
                'shared': {'centroid': (w * 0.5, h * 0.6), 'furniture': ['sofa']}
            }
    
    def get_centroid(self, item):
        """Calculate centroid (center point) of furniture item"""
        return (item['x'] + item['width'] / 2, 
                item['y'] + item['height'] / 2)
    
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + 
                        (point1[1] - point2[1])**2)
    
    def get_furniture_polygon(self, item):
        """Get Shapely polygon for furniture item (considering rotation)"""
        x, y = item['x'], item['y']
        w, h = item['width'], item['height']
        rect = box(x, y, x + w, y + h)
        
        rotation = item.get('rotation', 0)
        if rotation != 0:
            center = (x + w/2, y + h/2)
            rect = shapely_rotate(rect, rotation, origin=center)
        
        return rect
    
    # ========================================================================
    # COST COMPONENT 1: C_flow (Flow and Adjacency Cost)
    # ========================================================================
    
    def calculate_flow_cost(self):
        """
        C_flow = Σ(i,j) A_ij * d(c_i, c_j)²
        
        Minimizes distance between functionally related items.
        """
        total_cost = 0
        
        for i, item1 in enumerate(self.furniture):
            for item2 in self.furniture[i+1:]:
                # Get affinity score
                affinity = self._get_affinity(item1['name'], item2['name'])
                
                if affinity > 0:
                    # Calculate squared distance between centroids
                    c1 = self.get_centroid(item1)
                    c2 = self.get_centroid(item2)
                    distance = self.euclidean_distance(c1, c2)
                    
                    cost = affinity * (distance ** 2)
                    total_cost += cost
        
        return total_cost
    
    def _get_affinity(self, name1, name2):
        """Get affinity score between two furniture items"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Check affinity matrix
        for (key1, key2), affinity in self.AFFINITY_MATRIX.items():
            if (key1 in name1_lower and key2 in name2_lower) or \
               (key1 in name2_lower and key2 in name1_lower):
                return affinity
        
        return 0  # No affinity
    
    # ========================================================================
    # COST COMPONENT 2: C_zone (Zoning and Territory Cost)
    # ========================================================================
    
    def calculate_zone_cost(self):
        """
        C_zone = Σ(i∈Z1) d(c_i, μ1)² + Σ(j∈Z2) d(c_j, μ2)² + Σ(k∈ZS) d(c_k, μS)²
        
        Minimizes displacement from intended functional zones.
        """
        total_cost = 0
        
        for item in self.furniture:
            # Find which zone this furniture belongs to
            item_zone = self._get_item_zone(item['name'])
            
            if item_zone:
                zone_centroid = self.zones[item_zone]['centroid']
                item_centroid = self.get_centroid(item)
                
                # Squared distance from item to its zone center
                distance = self.euclidean_distance(item_centroid, zone_centroid)
                total_cost += distance ** 2
        
        return total_cost
    
    def _get_item_zone(self, furniture_name):
        """Determine which zone a furniture item belongs to"""
        name_lower = furniture_name.lower()
        
        for zone_name, zone_data in self.zones.items():
            for furniture_type in zone_data['furniture']:
                if furniture_type in name_lower:
                    return zone_name
        
        return None
    
    # ========================================================================
    # COST COMPONENT 3: C_env (Environmental Interaction Cost)
    # ========================================================================
    
    def calculate_environmental_cost(self):
        """
        C_env = Σ(k) P_window(f_k) + Σ(m) P_light(f_m)
        
        Penalizes poor interaction with windows and natural light.
        """
        window_cost = 0
        light_cost = 0
        
        windows = [o for o in self.openings if o['type'] == 'window']
        
        for item in self.furniture:
            # P_window: Penalty for tall furniture blocking windows
            if self._is_tall_furniture(item):
                for window in windows:
                    if self._blocks_window(item, window):
                        window_cost += self.PENALTY_WINDOW_OBSTRUCTION
            
            # P_light: Distance penalty for work surfaces from windows
            if self._needs_natural_light(item):
                if windows:
                    nearest_window = self._find_nearest_window(item, windows)
                    item_centroid = self.get_centroid(item)
                    window_centroid = self._get_window_centroid(nearest_window)
                    distance = self.euclidean_distance(item_centroid, window_centroid)
                    
                    # Normalize distance cost (penalize if > 200cm from window)
                    if distance > 200:
                        light_cost += (distance - 200) * 2
        
        return window_cost + light_cost
    
    def _is_tall_furniture(self, item):
        """Check if furniture is tall (>140cm)"""
        z_height = int(item.get('zHeight', 0))
        return z_height > 140
    
    def _needs_natural_light(self, item):
        """Check if furniture needs natural light (study table, desk)"""
        name_lower = item['name'].lower()
        return any(keyword in name_lower for keyword in ['study', 'desk', 'table'])
    
    def _blocks_window(self, item, window):
        """Check if furniture blocks window opening"""
        item_poly = self.get_furniture_polygon(item)
        window_zone = self._get_window_zone(window)
        return item_poly.intersects(window_zone)
    
    def _get_window_zone(self, window):
        """Get 2D footprint zone for window"""
        wall = window['wall']
        pos = window['position']
        size = window['size']
        
        clearance = 50  # 50cm buffer
        
        if wall == 'right':
            return box(self.room['width'] - clearance, pos, 
                      self.room['width'], pos + size)
        elif wall == 'left':
            return box(0, pos, clearance, pos + size)
        elif wall == 'top':
            return box(pos, 0, pos + size, clearance)
        else:  # bottom
            return box(pos, self.room['height'] - clearance, 
                      pos + size, self.room['height'])
    
    def _find_nearest_window(self, item, windows):
        """Find nearest window to furniture item"""
        item_centroid = self.get_centroid(item)
        min_distance = float('inf')
        nearest = windows[0]
        
        for window in windows:
            window_centroid = self._get_window_centroid(window)
            distance = self.euclidean_distance(item_centroid, window_centroid)
            if distance < min_distance:
                min_distance = distance
                nearest = window
        
        return nearest
    
    def _get_window_centroid(self, window):
        """Get centroid of window opening"""
        wall = window['wall']
        pos = window['position']
        size = window['size']
        
        if wall == 'right':
            return (self.room['width'], pos + size/2)
        elif wall == 'left':
            return (0, pos + size/2)
        elif wall == 'top':
            return (pos + size/2, 0)
        else:  # bottom
            return (pos + size/2, self.room['height'])
    
    # ========================================================================
    # COST COMPONENT 4: C_clearance (Ergonomic and Access Cost)
    # ========================================================================
    
    def calculate_clearance_cost(self):
        """
        C_clearance = Σ(d∈Doors) P_overlap(L, Z_swing_d) + Σ(c∈Chairs) P_overlap(L, Z_pullout_c)
        
        Very high penalty for blocking critical movement zones.
        """
        door_cost = 0
        chair_cost = 0
        
        # Check door swing zones
        doors = [o for o in self.openings if o['type'] == 'door']
        for door in doors:
            swing_zone = self._get_door_swing_zone(door)
            
            for item in self.furniture:
                item_poly = self.get_furniture_polygon(item)
                if swing_zone.intersects(item_poly):
                    door_cost += self.PENALTY_DOOR_SWING_BLOCK
        
        # Check chair pull-out zones
        chairs = [f for f in self.furniture if 'chair' in f['name'].lower()]
        for chair in chairs:
            pullout_zone = self._get_chair_pullout_zone(chair)
            
            for item in self.furniture:
                if item['name'] == chair['name']:
                    continue
                
                item_poly = self.get_furniture_polygon(item)
                if pullout_zone.intersects(item_poly):
                    chair_cost += self.PENALTY_CHAIR_PULLOUT_BLOCK
        
        return door_cost + chair_cost
    
    def _get_door_swing_zone(self, door):
        """Get door swing arc zone (90cm radius + buffer)"""
        wall = door['wall']
        pos = door['position']
        size = door['size']
        radius = self.DOOR_SWING_CLEARANCE + 50  # Add 50cm buffer
        
        if wall == 'right':
            center = (self.room['width'], pos + size/2)
            swing = Point(center).buffer(radius)
            # Clip to room interior
            room_clip = box(self.room['width'] - radius, 0, 
                           self.room['width'], self.room['height'])
            return swing.intersection(room_clip)
        
        elif wall == 'left':
            center = (0, pos + size/2)
            swing = Point(center).buffer(radius)
            room_clip = box(0, 0, radius, self.room['height'])
            return swing.intersection(room_clip)
        
        elif wall == 'top':
            center = (pos + size/2, 0)
            swing = Point(center).buffer(radius)
            room_clip = box(0, 0, self.room['width'], radius)
            return swing.intersection(room_clip)
        
        else:  # bottom
            center = (pos + size/2, self.room['height'])
            swing = Point(center).buffer(radius)
            room_clip = box(0, self.room['height'] - radius, 
                           self.room['width'], self.room['height'])
            return swing.intersection(room_clip)
    
    def _get_chair_pullout_zone(self, chair):
        """Get chair pull-out zone (80cm depth behind chair)"""
        x, y = chair['x'], chair['y']
        w, h = chair['width'], chair['height']
        rotation = chair.get('rotation', 0)
        depth = self.CHAIR_PULLOUT_DEPTH
        
        # Create pullout zone based on rotation
        if rotation == 0:
            zone = box(x, y + h, x + w, y + h + depth)
        elif rotation == 90:
            zone = box(x - depth, y, x, y + h)
        elif rotation == 180:
            zone = box(x, y - depth, x + w, y)
        else:  # 270
            zone = box(x + w, y, x + w + depth, y + h)
        
        return zone
    
    # ========================================================================
    # COST COMPONENT 5: C_vis (Aesthetics and Visibility Cost)
    # ========================================================================
    
    def calculate_aesthetics_cost(self):
        """
        C_vis = Σ(k) P_view_block(f_k) + Σ(l) P_central_placement(f_l)
        
        Subjective cost for visual quality and line-of-sight.
        """
        view_cost = 0
        placement_cost = 0
        
        # Find key viewing positions (e.g., sofa center)
        sofas = [f for f in self.furniture if 'sofa' in f['name'].lower()]
        windows = [o for o in self.openings if o['type'] == 'window']
        
        # P_view_block: Penalty for tall furniture blocking views
        if sofas and windows:
            for sofa in sofas:
                sofa_centroid = self.get_centroid(sofa)
                
                for window in windows:
                    window_centroid = self._get_window_centroid(window)
                    
                    # Check if any tall furniture blocks the line of sight
                    for item in self.furniture:
                        if self._is_tall_furniture(item) and item['name'] != sofa['name']:
                            if self._blocks_line_of_sight(sofa_centroid, window_centroid, item):
                                view_cost += self.PENALTY_VIEW_BLOCK
        
        # P_central_placement: Penalty for wall furniture in center
        for item in self.furniture:
            if self._should_be_against_wall(item):
                if self._is_in_room_center(item):
                    placement_cost += self.PENALTY_CENTRAL_PLACEMENT
        
        return view_cost + placement_cost
    
    def _blocks_line_of_sight(self, point1, point2, item):
        """Check if furniture blocks line of sight between two points"""
        from shapely.geometry import LineString
        
        line = LineString([point1, point2])
        item_poly = self.get_furniture_polygon(item)
        
        return line.intersects(item_poly)
    
    def _should_be_against_wall(self, item):
        """Check if furniture should typically be against wall"""
        name_lower = item['name'].lower()
        return any(keyword in name_lower for keyword in ['wardrobe', 'closet', 'shelf'])
    
    def _is_in_room_center(self, item):
        """Check if furniture is in room center (not near walls)"""
        wall_margin = 100  # 100cm from wall = "central"
        
        x, y = item['x'], item['y']
        w, h = item['width'], item['height']
        
        # Check if furniture centroid is in central zone
        centroid = self.get_centroid(item)
        
        in_center_x = wall_margin < centroid[0] < (self.room['width'] - wall_margin)
        in_center_y = wall_margin < centroid[1] < (self.room['height'] - wall_margin)
        
        return in_center_x and in_center_y
    
    # ========================================================================
    # TOTAL COST CALCULATION
    # ========================================================================
    
    def calculate_total_cost(self):
        """
        Calculate total weighted cost:
        C_total = w1*C_flow + w2*C_zone + w3*C_env + w4*C_clearance + w5*C_vis
        """
        costs = {
            'C_flow': self.calculate_flow_cost(),
            'C_zone': self.calculate_zone_cost(),
            'C_env': self.calculate_environmental_cost(),
            'C_clearance': self.calculate_clearance_cost(),
            'C_vis': self.calculate_aesthetics_cost()
        }
        
        # Apply weights
        weighted_costs = {
            'C_flow': costs['C_flow'] * self.WEIGHTS['w1'],
            'C_zone': costs['C_zone'] * self.WEIGHTS['w2'],
            'C_env': costs['C_env'] * self.WEIGHTS['w3'],
            'C_clearance': costs['C_clearance'] * self.WEIGHTS['w4'],
            'C_vis': costs['C_vis'] * self.WEIGHTS['w5']
        }
        
        total = sum(weighted_costs.values())
        
        return {
            'total': total,
            'components': costs,
            'weighted_components': weighted_costs,
            'weights': self.WEIGHTS
        }


# ============================================================================
# COMPARISON AND REPORTING
# ============================================================================

def compare_layouts(original_file, optimized_file):
    """
    Compare two layouts and generate detailed cost comparison report
    """
    
    print("="*80)
    print(" SCIENTIFIC COST FUNCTION VALIDATION")
    print(" Multi-Objective Facility Layout Optimization")
    print("="*80)
    
    # Load layouts
    try:
        with open(original_file, 'r') as f:
            original_layout = json.load(f)
        print(f"\nâœ… Loaded original: {original_file}")
    except Exception as e:
        print(f"\nâŒ Error loading original layout: {e}")
        return
    
    try:
        with open(optimized_file, 'r') as f:
            optimized_layout = json.load(f)
        print(f"âœ… Loaded optimized: {optimized_file}")
    except Exception as e:
        print(f"\nâŒ Error loading optimized layout: {e}")
        return
    
    # Calculate costs
    print("\n[*] Calculating cost functions...")
    
    original_validator = CostFunctionValidator(original_layout)
    original_costs = original_validator.calculate_total_cost()
    
    optimized_validator = CostFunctionValidator(optimized_layout)
    optimized_costs = optimized_validator.calculate_total_cost()
    
    # Generate report
    print("\n" + "="*80)
    print(" COST COMPARISON REPORT")
    print("="*80)
    
    print(f"\n[INFO] Weights Configuration:")
    for key, value in original_costs['weights'].items():
        component_name = {
            'w1': 'Flow/Adjacency',
            'w2': 'Zoning',
            'w3': 'Environmental',
            'w4': 'Clearance/Ergonomics',
            'w5': 'Aesthetics'
        }[key]
        print(f"   {key} ({component_name}): {value:.2f}")
    
    print(f"\n" + "-"*80)
    print(f"{'COST COMPONENT':<30} {'ORIGINAL':>15} {'OPTIMIZED':>15} {'IMPROVEMENT':>15}")
    print("-"*80)
    
    # Component-wise comparison
    for component in ['C_flow', 'C_zone', 'C_env', 'C_clearance', 'C_vis']:
        orig_val = original_costs['components'][component]
        opt_val = optimized_costs['components'][component]
        
        if orig_val > 0:
            improvement = ((orig_val - opt_val) / orig_val) * 100
        else:
            improvement = 0 if opt_val == 0 else -100
        
        component_name = {
            'C_flow': 'Flow & Adjacency',
            'C_zone': 'Zoning',
            'C_env': 'Environmental',
            'C_clearance': 'Clearance & Ergonomics',
            'C_vis': 'Aesthetics'
        }[component]
        
        symbol = "âœ…" if improvement > 0 else ("âš ï¸" if improvement < 0 else "â†'")
        print(f"{component_name:<30} {orig_val:>15.2f} {opt_val:>15.2f} {symbol} {improvement:>12.1f}%")
    
    print("-"*80)
    
    # Weighted costs
    print(f"\n{'WEIGHTED COSTS':<30} {'ORIGINAL':>15} {'OPTIMIZED':>15} {'IMPROVEMENT':>15}")
    print("-"*80)
    
    for component in ['C_flow', 'C_zone', 'C_env', 'C_clearance', 'C_vis']:
        orig_val = original_costs['weighted_components'][component]
        opt_val = optimized_costs['weighted_components'][component]
        
        if orig_val > 0:
            improvement = ((orig_val - opt_val) / orig_val) * 100
        else:
            improvement = 0 if opt_val == 0 else -100
        
        component_name = {
            'C_flow': 'Flow & Adjacency',
            'C_zone': 'Zoning',
            'C_env': 'Environmental',
            'C_clearance': 'Clearance & Ergonomics',
            'C_vis': 'Aesthetics'
        }[component]
        
        symbol = "âœ…" if improvement > 0 else ("âš ï¸" if improvement < 0 else "â†'")
        print(f"{component_name:<30} {orig_val:>15.2f} {opt_val:>15.2f} {symbol} {improvement:>12.1f}%")
    
    print("="*80)
    
    # Total cost comparison
    orig_total = original_costs['total']
    opt_total = optimized_costs['total']
    
    if orig_total > 0:
        total_improvement = ((orig_total - opt_total) / orig_total) * 100
    else:
        total_improvement = 0
    
    print(f"\nðŸŽ¯ TOTAL COST:")
    print(f"   Original:  {orig_total:>15.2f}")
    print(f"   Optimized: {opt_total:>15.2f}")
    print(f"   {'âœ…' if total_improvement > 0 else 'âŒ'} Overall Improvement: {total_improvement:>10.2f}%")
    
    # Conclusion
    print("\n" + "="*80)
    print(" CONCLUSION")
    print("="*80)
    
    if total_improvement > 20:
        verdict = "SIGNIFICANT IMPROVEMENT"
        emoji = "ðŸŽ‰"
    elif total_improvement > 5:
        verdict = "MODERATE IMPROVEMENT"
        emoji = "âœ…"
    elif total_improvement > 0:
        verdict = "MINOR IMPROVEMENT"
        emoji = "â†—ï¸"
    elif total_improvement == 0:
        verdict = "NO CHANGE"
        emoji = "â†'"
    else:
        verdict = "DEGRADATION"
        emoji = "âš ï¸"
    
    print(f"\n{emoji} {verdict}: {abs(total_improvement):.2f}% cost change")
    
    # Detailed analysis
    print(f"\n[STATS] Breakdown:")
    improvements = []
    degradations = []
    
    for component in ['C_flow', 'C_zone', 'C_env', 'C_clearance', 'C_vis']:
        orig_val = original_costs['components'][component]
        opt_val = optimized_costs['components'][component]
        
        if orig_val > 0:
            change = ((orig_val - opt_val) / orig_val) * 100
        else:
            change = 0 if opt_val == 0 else -100
        
        component_name = {
            'C_flow': 'Flow & Adjacency',
            'C_zone': 'Zoning',
            'C_env': 'Environmental',
            'C_clearance': 'Clearance & Ergonomics',
            'C_vis': 'Aesthetics'
        }[component]
        
        if change > 0:
            improvements.append((component_name, change))
        elif change < 0:
            degradations.append((component_name, abs(change)))
    
    if improvements:
        print(f"\n[OK] Improvements:")
        for name, change in sorted(improvements, key=lambda x: x[1], reverse=True):
            print(f"   â€¢ {name}: {change:.1f}% better")
    
    if degradations:
        print(f"\nâš ï¸ Degradations:")
        for name, change in sorted(degradations, key=lambda x: x[1], reverse=True):
            print(f"   â€¢ {name}: {change:.1f}% worse")
    
    print("\n" + "="*80)
    
    return {
        'original': original_costs,
        'optimized': optimized_costs,
        'improvement': total_improvement
    }


def main():
    """Main entry point"""
    
    original_file = "Input-Layouts/room-layout (11).json"
    optimized_file = "Output-Layouts/room-layout (11)-optimized.json"
    
    compare_layouts(original_file, optimized_file)


if __name__ == "__main__":
    main()
