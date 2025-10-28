"""
Room Layout Optimizer - Advanced Merged Version
Combines: 
- ViolationTracker with detailed reporting (Script 2)
- Hard overlap rejection (Script 1)
- Local optimization for fine-tuning (Script 1)
- Smart placement strategies (Script 2)
- 1cm grid precision (Script 1)
- Door clearance checking (Script 2)
- Comprehensive user feedback (Script 2)
"""

from validator import LayoutValidator
import copy
import random
import math
from shapely.geometry import box, Point
from shapely.affinity import rotate as shapely_rotate


class ViolationTracker:
    """Tracks violations throughout optimization"""
    
    def __init__(self):
        self.initial_violations = []
        self.final_violations = []
        
    def set_initial(self, layout):
        validator = LayoutValidator(layout)
        self.initial_violations = validator.validate()
        return self.initial_violations
    
    def finalize(self, final_layout):
        validator = LayoutValidator(final_layout)
        self.final_violations = validator.validate()
        
        initial_set = set(self.initial_violations)
        final_set = set(self.final_violations)
        
        fixed = list(initial_set - final_set)
        remaining = list(final_set)
        
        return {
            'initial_count': len(self.initial_violations),
            'final_count': len(self.final_violations),
            'fixed_count': len(fixed),
            'fixed': fixed,
            'remaining': remaining,
            'improvement': len(self.initial_violations) - len(self.final_violations)
        }
    
    def get_summary(self):
        """Categorize violations for UI"""
        
        def categorize(violations):
            categories = {
                'Overlaps': [],
                'Clearances': [],
                'Bed Clearances': [],
                'Turning Space': [],
                'Door': [],
                'Emergency Path': [],
                'Windows': [],
                'Heights': [],
                'Other': []
            }
            
            for v in violations:
                v_lower = v.lower()
                if "overlap" in v_lower:
                    categories['Overlaps'].append(v)
                elif "bed" in v_lower and "clearance" in v_lower:
                    categories['Bed Clearances'].append(v)
                elif "clearance" in v_lower:
                    categories['Clearances'].append(v)
                elif "turning" in v_lower:
                    categories['Turning Space'].append(v)
                elif "door" in v_lower:
                    categories['Door'].append(v)
                elif "emergency" in v_lower or "path" in v_lower:
                    categories['Emergency Path'].append(v)
                elif "window" in v_lower or "sill" in v_lower:
                    categories['Windows'].append(v)
                elif "height" in v_lower or "reach" in v_lower:
                    categories['Heights'].append(v)
                else:
                    categories['Other'].append(v)
            
            return {k: v for k, v in categories.items() if v}
        
        return {
            'initial': categorize(self.initial_violations),
            'fixed': categorize([v for v in self.initial_violations if v not in self.final_violations]),
            'remaining': categorize(self.final_violations)
        }


class LayoutOptimizer:
    """Advanced room layout optimizer with merged capabilities"""
    
    def __init__(self, input_layout):
        self.room = input_layout['room']
        self.furniture = input_layout['furniture']
        self.openings = input_layout['openings']
        self.tracker = ViolationTracker()
        self.unplaced_furniture = []
        
        # High precision grid (from Script 1)
        self.grid_step = 1
        
        # Furniture priority for removal if needed
        self.priority = ['Bed', 'Wardrobe', 'Bedside', 'Study', 'Sofa']
    
    def optimize(self, max_iterations=300, enable_local_optimization=True):
        """Main optimization with hard overlap rejection"""
        
        print(f"\n{'='*80}")
        print(f"ADVANCED ROOM LAYOUT OPTIMIZER")
        print(f"{'='*80}")
        print(f"Room: {self.room['width']}×{self.room['height']}cm")
        print(f"Furniture: {len(self.furniture)} items")
        print(f"Grid Precision: {self.grid_step}cm")
        print(f"Local Optimization: {'Enabled' if enable_local_optimization else 'Disabled'}")
        
        print(f"\n Analyzing initial layout...")
        self.tracker.set_initial({'room': self.room, 'furniture': self.furniture, 'openings': self.openings})
        print(f"   Initial violations: {len(self.tracker.initial_violations)}")
        
        door = next((o for o in self.openings if o['type'] == 'door'), None)
        if not door:
            door = {'wall': 'bottom', 'position': self.room['width']/2, 'size': 90}
        
        windows = [o for o in self.openings if o['type'] == 'window']
        
        print(f"\n  Generating layouts (hard overlap rejection enabled)...")
        
        best_layout = None
        best_score = float('inf')
        best_placed_count = 0
        attempts_without_overlap = 0
        
        for attempt in range(max_iterations):
            random.seed(attempt)
            
            candidate = self._generate_layout(door, windows)
            
            if not candidate or len(candidate['furniture']) == 0:
                continue
            
            # CRITICAL: Hard overlap rejection (from Script 1)
            if self._has_overlap(candidate):
                continue
            
            attempts_without_overlap += 1
            
            violations = self._count_violations(candidate)
            violation_count = len(violations)
            placed_count = len(candidate['furniture'])
            
            is_better = False
            if placed_count > best_placed_count:
                is_better = True
            elif placed_count == best_placed_count and violation_count < best_score:
                is_better = True
            
            if is_better:
                best_score = violation_count
                best_placed_count = placed_count
                best_layout = copy.deepcopy(candidate)
                
                if attempt % 25 == 0:
                    print(f"   Attempt {attempt} (valid: {attempts_without_overlap}): {placed_count}/{len(self.furniture)} items, {violation_count} violations, NO OVERLAPS ✓")
            
            # Perfect layout found
            if violation_count == 0 and placed_count == len(self.furniture):
                print(f"    Perfect layout at attempt {attempt}!")
                break
            
            # Local optimization every 50 iterations (from Script 1)
            if enable_local_optimization and attempt % 50 == 0 and best_layout:
                print(f"    Running local optimization...")
                optimized = self._local_optimize(best_layout, door)
                if not self._has_overlap(optimized):
                    optimized_violations = self._count_violations(optimized)
                    if len(optimized_violations) < best_score:
                        best_score = len(optimized_violations)
                        best_layout = optimized
                        print(f"      ✓ Improved to {best_score} violations")
        
        # GUARANTEE: Final overlap check (from Script 1)
        if best_layout and self._has_overlap(best_layout):
            print(f"\n  WARNING: Best layout has overlaps - regenerating...")
            return self.optimize(max_iterations=200, enable_local_optimization=False)
        
        if best_layout:
            summary = self.tracker.finalize(best_layout)
            
            print(f"\n{'='*80}")
            print(f"OPTIMIZATION COMPLETE")
            print(f"{'='*80}")
            print(f"✓ Furniture placed: {len(best_layout['furniture'])}/{len(self.furniture)}")
            print(f"✓ Violations: {summary['initial_count']} → {summary['final_count']}")
            print(f"✓ Fixed: {summary['fixed_count']} violations")
            print(f"✓ Improvement: {summary['improvement']}")
            print(f"✓ Valid attempts: {attempts_without_overlap}")
            print(f"✓ Overlaps: ZERO (guaranteed) ✓")
            
            if len(best_layout['furniture']) < len(self.furniture):
                self._analyze_unplaced(best_layout)
            
            return best_layout
        else:
            print(f"\n✗ No valid layout found")
            return None
    
    def _local_optimize(self, layout, door):
        """Local optimization with small adjustments (from Script 1)"""
        improved = copy.deepcopy(layout)
        current_violations = len(self._count_violations(improved))
        
        # Try small position adjustments for each item
        for item in improved['furniture']:
            # Skip paired items (chairs follow tables)
            if 'chair' in item['name'].lower():
                continue
            
            original_x, original_y = item['x'], item['y']
            best_item_x, best_item_y = original_x, original_y
            best_item_violations = current_violations
            
            # Try moving ±50cm, ±20cm, ±10cm in each direction
            for dx in [-50, -20, -10, 0, 10, 20, 50]:
                for dy in [-50, -20, -10, 0, 10, 20, 50]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    item['x'] = max(0, min(original_x + dx, 
                                self.room['width'] - item['width']))
                    item['y'] = max(0, min(original_y + dy, 
                                self.room['height'] - item['height']))
                    
                    # Check if valid (no overlaps, not blocking door)
                    if not self._has_overlap(improved) and not self._blocks_door(item, door):
                        violations = self._count_violations(improved)
                        if len(violations) < best_item_violations:
                            best_item_violations = len(violations)
                            best_item_x, best_item_y = item['x'], item['y']
                            
                            # Early exit if perfect
                            if best_item_violations == 0:
                                return improved
            
            # Apply best position for this item
            item['x'], item['y'] = best_item_x, best_item_y
            current_violations = best_item_violations
        
        return improved
    
    def _has_overlap(self, layout):
        """Check if any furniture overlaps (from Script 1)"""
        polygons = []
        for item in layout['furniture']:
            poly = self._get_polygon(item)
            
            # Check against all existing
            for existing_poly in polygons:
                if poly.intersects(existing_poly):
                    return True
            
            polygons.append(poly)
        
        return False
    
    def _generate_layout(self, door, windows):
        """Generate single layout with smart placement (from Script 2)"""
        
        layout = {
            'room': self.room,
            'furniture': [],
            'openings': self.openings
        }
        
        # Identify furniture types
        bed = next((f for f in self.furniture if 'bed' in f['name'].lower() and 'bedside' not in f['name'].lower()), None)
        bedsides = [f for f in self.furniture if 'bedside' in f['name'].lower()]
        wardrobes = [f for f in self.furniture if 'wardrobe' in f['name'].lower()]
        tables = [f for f in self.furniture if 'table' in f['name'].lower() and 'bedside' not in f['name'].lower()]
        chairs = [f for f in self.furniture if 'chair' in f['name'].lower()]
        sofas = [f for f in self.furniture if 'sofa' in f['name'].lower()]
        
        # STEP 1: Place bed FIRST
        if bed:
            placed_bed = self._place_bed(copy.deepcopy(bed), door, layout)
            if placed_bed:
                layout['furniture'].append(placed_bed)
                
                # STEP 2: Place bedsides IMMEDIATELY
                for bedside in bedsides:
                    placed_bedside = self._place_bedside_near_bed(copy.deepcopy(bedside), placed_bed, layout)
                    if placed_bedside:
                        layout['furniture'].append(placed_bedside)
        
        # STEP 3: Place wardrobes
        for wardrobe in wardrobes:
            placed = self._place_wardrobe(copy.deepcopy(wardrobe), door, layout)
            if placed:
                layout['furniture'].append(placed)
        
        # STEP 4: Place tables
        for table in tables:
            if 'study' in table['name'].lower():
                placed = self._place_near_window(copy.deepcopy(table), windows, layout, door)
                # CHANGE: Don't add if it couldn't be placed near window
                if placed:
                    layout['furniture'].append(placed)
                    
                    # Place associated chairs immediately
                    for chair in chairs:
                        if 'study' in chair['name'].lower():
                            placed_chair = self._place_chair_at_table(copy.deepcopy(chair), placed, layout, door)
                            if placed_chair:
                                layout['furniture'].append(placed_chair)
                                chairs.remove(chair)
                                break
            else:
                placed = self._place_along_wall(copy.deepcopy(table), layout, door)
                if placed:
                    layout['furniture'].append(placed)
        # STEP 5: Place sofas
        for sofa in sofas:
            placed = self._place_along_wall(copy.deepcopy(sofa), layout, door)
            if placed:
                layout['furniture'].append(placed)
        
        # STEP 6: Place remaining chairs
        for chair in chairs:
            placed = self._place_anywhere_grid(copy.deepcopy(chair), layout, door)
            if placed:
                layout['furniture'].append(placed)
        
        return layout
    
    def _place_bed(self, bed, door, layout):
        """Place bed against wall away from door (high precision)"""
        door_wall = door['wall'] if door else None
        
        walls = ['left', 'right', 'top', 'bottom']
        valid_walls = [w for w in walls if w != door_wall]
        
        if not valid_walls:
            valid_walls = walls
        
        random.shuffle(valid_walls)
        
        for wall in valid_walls:
            if wall == 'left':
                bed['x'] = 10
                bed['y'] = random.randrange(50, max(51, self.room['height'] - bed['height'] - 50), 
                                           self.grid_step)
                bed['rotation'] = 0
            elif wall == 'right':
                bed['x'] = self.room['width'] - bed['width'] - 10
                bed['y'] = random.randrange(50, max(51, self.room['height'] - bed['height'] - 50), 
                                           self.grid_step)
                bed['rotation'] = 0
            elif wall == 'top':
                bed['x'] = random.randrange(50, max(51, self.room['width'] - bed['width'] - 50), 
                                           self.grid_step)
                bed['y'] = 10
                bed['rotation'] = 0
            else:  # bottom
                bed['x'] = random.randrange(50, max(51, self.room['width'] - bed['width'] - 50), 
                                           self.grid_step)
                bed['y'] = self.room['height'] - bed['height'] - 10
                bed['rotation'] = 0
            
            if self._check_bounds(bed) and self._is_valid(bed, layout, door):
                return bed
        
        return None
    
    def _place_bedside_near_bed(self, bedside, bed, layout):
        """Place bedside table near bed head with precision"""
        rot = bed.get('rotation', 0)
        gap = 5
        
        # Get actual bed dimensions
        if rot in [0, 180]:
            bed_w, bed_h = bed['width'], bed['height']
        else:
            bed_w, bed_h = bed['height'], bed['width']
        
        # Try positions at bed head
        positions = []
        
        if rot == 0:
            # Right side of bed head
            positions.append((bed['x'] + bed_w + gap, bed['y'], 0))
            # Left side of bed head
            positions.append((bed['x'] - bedside['width'] - gap, bed['y'], 0))
        elif rot == 180:
            # Right side
            positions.append((bed['x'] + bed_w + gap, bed['y'] + bed_h - bedside['height'], 0))
            # Left side
            positions.append((bed['x'] - bedside['width'] - gap, bed['y'] + bed_h - bedside['height'], 0))
        elif rot == 90:
            # Top
            positions.append((bed['x'], bed['y'] - bedside['height'] - gap, 0))
            # Bottom
            positions.append((bed['x'] + bed_w - bedside['width'], bed['y'] - bedside['height'] - gap, 0))
        else:  # 270
            # Top
            positions.append((bed['x'], bed['y'] + bed_h + gap, 0))
            # Bottom
            positions.append((bed['x'] + bed_w - bedside['width'], bed['y'] + bed_h + gap, 0))
        
        for x, y, r in positions:
            bedside['x'] = x
            bedside['y'] = y
            bedside['rotation'] = r
            
            if self._check_bounds(bedside) and self._is_valid(bedside, layout, None):
                return bedside
        
        return None
    
    def _place_wardrobe(self, wardrobe, door, layout):
        """Place wardrobe on wall away from door"""
        door_wall = door['wall'] if door else None
        
        walls = ['top', 'bottom', 'left', 'right']
        valid_walls = [w for w in walls if w != door_wall]
        
        if not valid_walls:
            valid_walls = walls
        
        random.shuffle(valid_walls)
        
        for wall in valid_walls:
            positions = self._get_wall_positions(wardrobe, wall, count=5)
            
            for x, y, rot in positions:
                wardrobe['x'] = x
                wardrobe['y'] = y
                wardrobe['rotation'] = rot
                
                if self._check_bounds(wardrobe) and self._is_valid(wardrobe, layout, door):
                    return wardrobe
        
        return None
    
    def _place_near_window(self, furniture, windows, layout, door):
        """Place furniture near window - PRIORITIZE WINDOW PLACEMENT"""
        if not windows:
            return self._place_along_wall(furniture, layout, door)
        
        # TRY ALL WINDOWS, not just one random window
        for window in windows:
            wall = window['wall']
            pos = window['position']
            
            # Try multiple distances from window (30cm, 50cm, 80cm, 120cm)
            for margin in [30, 50, 80, 120]:
                positions = []
                
                if wall == 'bottom':
                    # Try centered, left-offset, and right-offset positions
                    positions.append((pos - furniture['width']//2, self.room['height'] - furniture['height'] - margin, 180))
                    positions.append((pos - furniture['width'] - 20, self.room['height'] - furniture['height'] - margin, 180))
                    positions.append((pos + 20, self.room['height'] - furniture['height'] - margin, 180))
                elif wall == 'top':
                    positions.append((pos - furniture['width']//2, margin, 0))
                    positions.append((pos - furniture['width'] - 20, margin, 0))
                    positions.append((pos + 20, margin, 0))
                elif wall == 'left':
                    positions.append((margin, pos - furniture['height']//2, 270))
                    positions.append((margin, pos - furniture['height'] - 20, 270))
                    positions.append((margin, pos + 20, 270))
                else:  # right
                    positions.append((self.room['width'] - furniture['width'] - margin, pos - furniture['height']//2, 90))
                    positions.append((self.room['width'] - furniture['width'] - margin, pos - furniture['height'] - 20, 90))
                    positions.append((self.room['width'] - furniture['width'] - margin, pos + 20, 90))
                
                # Try all positions for this window at this distance
                for x, y, rot in positions:
                    furniture['x'] = max(0, min(x, self.room['width'] - furniture['width']))
                    furniture['y'] = max(0, min(y, self.room['height'] - furniture['height']))
                    furniture['rotation'] = rot
                    
                    if self._check_bounds(furniture) and self._is_valid(furniture, layout, door):
                        return furniture  # SUCCESS - placed near window!
        
        # Only fall back if NO window position worked at all
        return None  # Return None instead of fallback - force it to try window placement
    
    def _place_along_wall(self, furniture, layout, door):
        """Place furniture along any available wall"""
        walls = ['top', 'bottom', 'left', 'right']
        random.shuffle(walls)
        
        for wall in walls:
            positions = self._get_wall_positions(furniture, wall, count=5)
            
            for x, y, rot in positions:
                furniture['x'] = x
                furniture['y'] = y
                furniture['rotation'] = rot
                
                if self._check_bounds(furniture) and self._is_valid(furniture, layout, door):
                    return furniture
        
        return None
    
    def _place_chair_at_table(self, chair, table, layout, door):
        """Place chair at table"""
        rot = table.get('rotation', 0)
        gap = 10
        
        positions = []
        
        if rot == 0:
            positions.append((table['x'] + table['width']//2 - chair['width']//2, 
                            table['y'] + table['height'] + gap, 0))
        elif rot == 180:
            positions.append((table['x'] + table['width']//2 - chair['width']//2, 
                            table['y'] - chair['height'] - gap, 180))
        elif rot == 90:
            positions.append((table['x'] - chair['width'] - gap, 
                            table['y'] + table['height']//2 - chair['height']//2, 90))
        else:  # 270
            positions.append((table['x'] + table['width'] + gap, 
                            table['y'] + table['height']//2 - chair['height']//2, 270))
        
        for x, y, r in positions:
            chair['x'] = x
            chair['y'] = y
            chair['rotation'] = r
            
            if self._check_bounds(chair) and self._is_valid(chair, layout, door):
                return chair
        
        return None
    
    def _place_anywhere_grid(self, furniture, layout, door):
        """Place anywhere using fine grid search"""
        step = max(20, self.grid_step * 10)  # Adaptive step size
        
        for y in range(30, self.room['height'] - furniture['height'] - 30, step):
            for x in range(30, self.room['width'] - furniture['width'] - 30, step):
                furniture['x'] = x
                furniture['y'] = y
                furniture['rotation'] = random.choice([0, 90, 180, 270])
                
                if self._check_bounds(furniture) and self._is_valid(furniture, layout, door):
                    return furniture
        
        return None
    
    def _get_wall_positions(self, item, wall, count=5):
        """Get positions along a wall"""
        positions = []
        margin = 30
        
        if wall == 'top':
            step = max(50, (self.room['width'] - 2*margin - item['width']) // count)
            for i in range(count):
                x = margin + i * step
                if x + item['width'] <= self.room['width'] - margin:
                    positions.append((x, margin, 0))
        
        elif wall == 'bottom':
            step = max(50, (self.room['width'] - 2*margin - item['width']) // count)
            for i in range(count):
                x = margin + i * step
                if x + item['width'] <= self.room['width'] - margin:
                    positions.append((x, self.room['height'] - item['height'] - margin, 180))
        
        elif wall == 'left':
            step = max(50, (self.room['height'] - 2*margin - item['height']) // count)
            for i in range(count):
                y = margin + i * step
                if y + item['height'] <= self.room['height'] - margin:
                    positions.append((margin, y, 270))
        
        else:  # right
            step = max(50, (self.room['height'] - 2*margin - item['height']) // count)
            for i in range(count):
                y = margin + i * step
                if y + item['height'] <= self.room['height'] - margin:
                    positions.append((self.room['width'] - item['width'] - margin, y, 90))
        
        return positions
    
    def _blocks_door(self, furniture, door):
        """Check if blocks door - 120cm clearance"""
        if not door:
            return False
        
        wall = door['wall']
        door_pos = door['position']
        door_size = door['size']
        CLEARANCE = 120
        
        if wall == 'right':
            door_area_x = self.room['width'] - CLEARANCE
            door_area_y_start = door_pos - 30
            door_area_y_end = door_pos + door_size + 30
            
            if (furniture['x'] + furniture['width'] > door_area_x and
                furniture['y'] < door_area_y_end and
                furniture['y'] + furniture['height'] > door_area_y_start):
                return True
        
        elif wall == 'left':
            door_area_x_end = CLEARANCE
            door_area_y_start = door_pos - 30
            door_area_y_end = door_pos + door_size + 30
            
            if (furniture['x'] < door_area_x_end and
                furniture['y'] < door_area_y_end and
                furniture['y'] + furniture['height'] > door_area_y_start):
                return True
        
        elif wall == 'top':
            door_area_y_end = CLEARANCE
            door_area_x_start = door_pos - 30
            door_area_x_end = door_pos + door_size + 30
            
            if (furniture['y'] < door_area_y_end and
                furniture['x'] < door_area_x_end and
                furniture['x'] + furniture['width'] > door_area_x_start):
                return True
        
        else:  # bottom
            door_area_y = self.room['height'] - CLEARANCE
            door_area_x_start = door_pos - 30
            door_area_x_end = door_pos + door_size + 30
            
            if (furniture['y'] + furniture['height'] > door_area_y and
                furniture['x'] < door_area_x_end and
                furniture['x'] + furniture['width'] > door_area_x_start):
                return True
        
        return False
    
    def _analyze_unplaced(self, layout):
        """Analyze unplaced furniture"""
        placed_items = layout['furniture']
        input_ids = [f"{f['name']}_{i}_{f['width']}_{f['height']}" for i, f in enumerate(self.furniture)]
        placed_ids = []
        
        for placed in placed_items:
            for i, original in enumerate(self.furniture):
                item_id = f"{original['name']}_{i}_{original['width']}_{original['height']}"
                if (placed['name'] == original['name'] and 
                    placed['width'] == original['width'] and 
                    placed['height'] == original['height'] and
                    item_id not in placed_ids):
                    placed_ids.append(item_id)
                    break
        
        unplaced_ids = [id for id in input_ids if id not in placed_ids]
        self.unplaced_furniture = []
        
        for unplaced_id in unplaced_ids:
            idx = int(unplaced_id.split('_')[1])
            self.unplaced_furniture.append(self.furniture[idx])
        
        if self.unplaced_furniture:
            print(f"\n{'='*80}")
            print(f"  FURNITURE PLACEMENT SUGGESTIONS")
            print(f"{'='*80}")
            print(f"\nCouldn't place {len(self.unplaced_furniture)} item(s):")
            
            for item in self.unplaced_furniture:
                print(f"   {item['name']} ({item['width']}×{item['height']}cm)")
                
            print(f"\nSuggestions:")
            print(f"  • Consider removing some furniture")
            print(f"  • Try a larger room")
            print(f"  • Adjust furniture dimensions")
    
    def get_violation_report(self):
        """Get violation report for UI"""
        summary = self.tracker.get_summary()
        
        return {
            'initial': summary['initial'],
            'fixed': summary['fixed'],
            'remaining': summary['remaining'],
            'unplaced_furniture': self.unplaced_furniture
        }
    
    def _check_bounds(self, item):
        """Check if within room bounds"""
        return (item['x'] >= 0 and item['y'] >= 0 and
                item['x'] + item['width'] <= self.room['width'] and
                item['y'] + item['height'] <= self.room['height'])
    
    def _is_valid(self, item, layout, door):
        """Check if valid placement"""
        # Check door blocking (skip for bedsides)
        if door and 'bedside' not in item['name'].lower():
            if self._blocks_door(item, door):
                return False
        
        # Check overlaps
        item_poly = self._get_polygon(item)
        
        for other in layout['furniture']:
            other_poly = self._get_polygon(other)
            if item_poly.intersects(other_poly):
                return False
        
        return True
    
    def _get_polygon(self, item):
        """Get furniture polygon with rotation support"""
        x, y = item['x'], item['y']
        w, h = item['width'], item['height']
        rect = box(x, y, x + w, y + h)
        
        rotation = item.get('rotation', 0)
        if rotation != 0:
            center = (x + w/2, y + h/2)
            rect = shapely_rotate(rect, rotation, origin=center)
        
        return rect
    
    def _count_violations(self, layout):
        """Count violations"""
        validator = LayoutValidator(layout)
        return validator.validate()


# ===================== EXAMPLE USAGE =====================
if __name__ == "__main__":
    import json
    
    # Example layout
    example_layout = {
        'room': {
            'width': 400,
            'height': 350
        },
        'furniture': [
            {'name': 'Bed', 'width': 200, 'height': 140, 'x': 0, 'y': 0},
            {'name': 'Bedside Table 1', 'width': 50, 'height': 40, 'x': 0, 'y': 0},
            {'name': 'Bedside Table 2', 'width': 50, 'height': 40, 'x': 0, 'y': 0},
            {'name': 'Wardrobe', 'width': 180, 'height': 60, 'x': 0, 'y': 0},
            {'name': 'Study Table', 'width': 120, 'height': 60, 'x': 0, 'y': 0},
            {'name': 'Study Chair', 'width': 50, 'height': 50, 'x': 0, 'y': 0}
        ],
        'openings': [
            {'type': 'door', 'wall': 'bottom', 'position': 150, 'size': 90},
            {'type': 'window', 'wall': 'right', 'position': 100, 'size': 120}
        ]
    }
    
    # Run optimizer
    optimizer = LayoutOptimizer(example_layout)
    best_layout = optimizer.optimize(max_iterations=300, enable_local_optimization=True)
    
    if best_layout:
        print(f"\n{'='*80}")
        print(f"FINAL LAYOUT")
        print(f"{'='*80}")
        for item in best_layout['furniture']:
            print(f"{item['name']:20} @ ({item['x']:3.0f}, {item['y']:3.0f}) rot={item.get('rotation', 0):3.0f}°")
        
        # Get detailed report
        report = optimizer.get_violation_report()
        
        if report['remaining']:
            print(f"\n{'='*80}")
            print(f"REMAINING VIOLATIONS BY CATEGORY")
            print(f"{'='*80}")
            for category, violations in report['remaining'].items():
                print(f"\n{category}:")
                for v in violations:
                    print(f"  • {v}")