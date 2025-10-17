from validator import LayoutValidator
import copy
import random

class LayoutOptimizer:
    """Optimizes bedroom layouts for DIN 18040-2 compliance"""
    
    def __init__(self, input_layout):
        self.original = input_layout
        self.room = input_layout['room']
        self.furniture = input_layout['furniture']
        self.openings = input_layout['openings']
        
        # Grid resolution (cm)
        self.grid_step = 1
        
        # Furniture priority (remove in reverse order)
        self.priority = ['Bed', 'Wardrobe', 'Bedside', 'Study', 'Sofa']
    
    def optimize(self, max_iterations=5000):
        """Main optimization loop - NO OVERLAPS ALLOWED"""
        
        print(f"Optimizing with all furniture ({len(self.furniture)} items)...")
        
        best_layout = None
        best_score = float('inf')
        attempts_without_overlap = 0
        
        for iteration in range(max_iterations):
            candidate = self._generate_candidate(self.furniture)
            
            # CRITICAL: Skip if any overlap
            if self._has_overlap(candidate):
                continue
            
            attempts_without_overlap += 1
            
            violations = self._count_violations(candidate)
            score = len(violations)
            
            if score < best_score:
                best_score = score
                best_layout = copy.deepcopy(candidate)
                print(f"Iteration {iteration} (valid: {attempts_without_overlap}): {score} violations")
            
            if score == 0:
                print(f"OK Perfect layout at iteration {iteration}")
                return best_layout
            
            # Local optimization
            if iteration % 100 == 0 and best_layout:
                optimized = self._local_optimize(best_layout)
                if not self._has_overlap(optimized):
                    violations = self._count_violations(optimized)
                    if len(violations) < best_score:
                        best_score = len(violations)
                        best_layout = optimized
                        print(f"  Local opt: {best_score} violations")
        
        # GUARANTEE: Final check
        if best_layout and self._has_overlap(best_layout):
            print("WARNING WARNING: Best layout has overlaps - regenerating...")
            return self.optimize(max_iterations=1000)
        
        print(f"OK Final: {best_score} violations, ZERO overlaps")
        return best_layout

    def _local_optimize(self, layout):
        """Small adjustments to reduce violations"""
        improved = copy.deepcopy(layout)
        
        # Try small position adjustments
        for item in improved['furniture']:
            # Skip paired items (chair follows table)
            if 'chair' in item['name'].lower():
                continue
            
            # Try moving ±50cm in each direction
            original_x, original_y = item['x'], item['y']
            
            for dx in [-50, -20, 0, 20, 50]:
                for dy in [-50, -20, 0, 20, 50]:
                    item['x'] = max(0, min(original_x + dx, 
                                self.room['width'] - item['width']))
                    item['y'] = max(0, min(original_y + dy, 
                                self.room['height'] - item['height']))
                    
                    # Check if better
                    violations = self._count_violations(improved)
                    if len(violations) == 0:
                        return improved
            
            # Restore if no improvement
            item['x'], item['y'] = original_x, original_y
        
        return improved
    
    def _has_overlap(self, layout):
        """Check if any furniture overlaps"""
        from shapely.geometry import box
        
        polygons = []
        for item in layout['furniture']:
            x, y = item['x'], item['y']
            w, h = item['width'], item['height']
            poly = box(x, y, x + w, y + h)
            
            # Check against all existing
            for existing_poly in polygons:
                if poly.intersects(existing_poly):
                    return True
            
            polygons.append(poly)
        
        return False

    def _place_without_overlap(self, item, layout, max_attempts=50):
        """Try to place item without overlapping existing furniture"""
        
        for attempt in range(max_attempts):
            # Randomize position
            item['x'] = random.randrange(0, max(1, self.room['width'] - item['width']), 
                                        self.grid_step)
            item['y'] = random.randrange(0, max(1, self.room['height'] - item['height']), 
                                        self.grid_step)
            
            # Check if overlaps
            test_layout = copy.deepcopy(layout)
            test_layout['furniture'].append(item)
            
            if not self._has_overlap(test_layout):
                return item
        
        # Fallback: return anyway (will be caught by validator)
        return item
    
    def _optimize_positions(self, furniture_list, max_iter):
        """Optimize positions for given furniture list"""
        
        best_layout = None
        best_score = float('inf')
        
        for iteration in range(max_iter):
            # Generate candidate
            candidate = self._generate_candidate(furniture_list)
            
            # HARD REJECT: Skip if overlaps exist
            if self._has_overlap(candidate):
                continue
            
            # Evaluate
            violations = self._count_violations(candidate)
            score = len(violations)
            
            if score < best_score:
                best_score = score
                best_layout = copy.deepcopy(candidate)
                print(f"Iteration {iteration}: {score} violations (no overlaps)")
            
            # Stop if perfect
            if score == 0:
                print(f"OK Valid layout found at iteration {iteration}")
                return best_layout
        
        return best_layout
    
    def _generate_candidate(self, furniture_list):
        """Generate candidate with intelligent placement"""
        layout = {
            'room': self.room,
            'furniture': [],
            'openings': self.openings
        }
        
        # Sort furniture by placement priority
        sorted_furniture = self._sort_by_type(furniture_list)
        
        # Place each furniture type intelligently
        for item in sorted_furniture:
            new_item = copy.deepcopy(item)
            
            if 'bed' in item['name'].lower():
                new_item = self._place_bed(new_item, layout)
            elif 'wardrobe' in item['name'].lower():
                new_item = self._place_wardrobe(new_item, layout)
            elif 'bedside' in item['name'].lower():
                new_item = self._place_bedside_table(new_item, layout)
            elif 'study table' in item['name'].lower():
                new_item = self._place_study_table(new_item, layout)
            elif 'study chair' in item['name'].lower():
                new_item = self._place_study_chair(new_item, layout)
            elif 'sofa' in item['name'].lower():
                new_item = self._place_sofa(new_item, layout)
            else:
                new_item = self._place_generic(new_item, layout)
            
            layout['furniture'].append(new_item)
        
        return layout
    
    def _sort_by_type(self, furniture_list):
        """Sort furniture by placement order"""
        # CHANGED: Bedside comes BEFORE wardrobe
        order = ['bed', 'bedside', 'wardrobe', 'study table', 'study chair', 'sofa']
        
        def priority_key(item):
            name_lower = item['name'].lower()
            for i, keyword in enumerate(order):
                if keyword in name_lower:
                    return i
            return len(order)
        
        return sorted(furniture_list, key=priority_key)
    
    def _place_bed(self, bed, layout):
        """Place bed in commanding position"""
        door = next((o for o in self.openings if o['type'] == 'door'), None)
        window = next((o for o in self.openings if o['type'] == 'window'), None)
        
        # Find walls (avoid door wall)
        door_wall = door['wall'] if door else None
        window_wall = window['wall'] if window else None
        
        # Choose wall for headboard (not door wall, not window wall)
        walls = ['top', 'bottom', 'left', 'right']
        valid_walls = [w for w in walls if w != door_wall and w != window_wall]
        
        if not valid_walls:
            valid_walls = [w for w in walls if w != door_wall]
        
        chosen_wall = random.choice(valid_walls) if valid_walls else 'bottom'
        
        # Position against chosen wall
        if chosen_wall == 'top':
            bed['x'] = random.randrange(50, max(51, self.room['width'] - bed['width'] - 50), 
                                       self.grid_step)
            bed['y'] = 10
            bed['rotation'] = 0
        elif chosen_wall == 'bottom':
            bed['x'] = random.randrange(50, max(51, self.room['width'] - bed['width'] - 50), 
                                       self.grid_step)
            bed['y'] = self.room['height'] - bed['height'] - 10
            bed['rotation'] = 180
        elif chosen_wall == 'left':
            bed['x'] = 10
            bed['y'] = random.randrange(50, max(51, self.room['height'] - bed['width'] - 50), 
                                       self.grid_step)
            bed['rotation'] = 90
        else:  # right
            bed['x'] = self.room['width'] - bed['height'] - 10
            bed['y'] = random.randrange(50, max(51, self.room['height'] - bed['width'] - 50), 
                                       self.grid_step)
            bed['rotation'] = 270
        
        return bed
    
    def _place_wardrobe(self, wardrobe, layout):
        """Place wardrobe on side/back wall, away from door"""
        door = next((o for o in self.openings if o['type'] == 'door'), None)
        door_wall = door['wall'] if door else None
        
        # Choose wall (not door wall)
        walls = ['top', 'bottom', 'left', 'right']
        valid_walls = [w for w in walls if w != door_wall]
        
        chosen_wall = random.choice(valid_walls) if valid_walls else 'left'
        
        # Position against wall
        if chosen_wall == 'top':
            wardrobe['x'] = random.randrange(0, max(1, self.room['width'] - wardrobe['width']), 
                                            self.grid_step)
            wardrobe['y'] = 10
            wardrobe['rotation'] = 0
        elif chosen_wall == 'bottom':
            wardrobe['x'] = random.randrange(0, max(1, self.room['width'] - wardrobe['width']), 
                                            self.grid_step)
            wardrobe['y'] = self.room['height'] - wardrobe['height'] - 10
            wardrobe['rotation'] = 0
        elif chosen_wall == 'left':
            wardrobe['x'] = 10
            wardrobe['y'] = random.randrange(0, max(1, self.room['height'] - wardrobe['height']), 
                                            self.grid_step)
            wardrobe['rotation'] = 0
        else:  # right
            wardrobe['x'] = self.room['width'] - wardrobe['width'] - 10
            wardrobe['y'] = random.randrange(0, max(1, self.room['height'] - wardrobe['height']), 
                                            self.grid_step)
            wardrobe['rotation'] = 0
        
        return wardrobe
    
    def _place_bedside_table(self, table, layout):
        """Place bedside table directly adjacent to bed"""
        bed = next((f for f in layout['furniture'] if 'bed' in f['name'].lower()), None)
        
        if not bed:
            return self._place_generic(table, layout)
        
        rot = bed.get('rotation', 0)
        gap = 5
        
        # Get actual bed dimensions considering rotation
        if rot in [0, 180]:
            bed_w, bed_h = bed['width'], bed['height']
        else:  # 90, 270
            bed_w, bed_h = bed['height'], bed['width']
        
        # Place at bed head based on rotation
        if rot == 0:
            table['x'] = bed['x'] + bed_w + gap
            table['y'] = bed['y']
        elif rot == 180:
            table['x'] = bed['x'] + bed_w + gap
            table['y'] = bed['y'] + bed_h - table['height']
        elif rot == 90:
            table['x'] = bed['x']
            table['y'] = bed['y'] - table['height'] - gap
        else:  # 270
            table['x'] = bed['x'] + bed_w - table['width']
            table['y'] = bed['y'] - table['height'] - gap
        
        # Fallback if out of bounds
        if (table['x'] < 0 or table['x'] + table['width'] > self.room['width'] or
            table['y'] < 0 or table['y'] + table['height'] > self.room['height']):
            if rot == 0 or rot == 180:
                table['x'] = bed['x'] - table['width'] - gap
                table['y'] = bed['y']
            else:
                table['x'] = bed['x']
                table['y'] = bed['y'] + bed_h + gap
        
        # Final bounds check
        table['x'] = max(0, min(table['x'], self.room['width'] - table['width']))
        table['y'] = max(0, min(table['y'], self.room['height'] - table['height']))
        
        table['rotation'] = 0
        return table
    
    def _place_study_table(self, table, layout):
        """Place study table near window"""
        window = next((o for o in self.openings if o['type'] == 'window'), None)
        
        if window:
            wall = window['wall']
            pos = window['position']
            margin = 30  # Close to window, not too far
            
            # Place facing window
            if wall == 'bottom':
                table['x'] = pos - table['width']//2
                table['y'] = self.room['height'] - table['height'] - margin
                table['rotation'] = 180
            elif wall == 'top':
                table['x'] = pos - table['width']//2
                table['y'] = margin
                table['rotation'] = 0
            elif wall == 'left':
                table['x'] = margin
                table['y'] = pos - table['height']//2
                table['rotation'] = 270
            else:  # right
                table['x'] = self.room['width'] - table['width'] - margin
                table['y'] = pos - table['height']//2
                table['rotation'] = 90
            
            # Clamp to room bounds
            table['x'] = max(0, min(table['x'], self.room['width'] - table['width']))
            table['y'] = max(0, min(table['y'], self.room['height'] - table['height']))
            
        else:
            table = self._place_generic(table, layout)
        
        return table
    
    def _place_study_chair(self, chair, layout):
        """Place study chair at study table"""
        table = next((f for f in layout['furniture'] 
                     if 'study table' in f['name'].lower()), None)
        
        if not table:
            return self._place_generic(chair, layout)
        
        rot = table.get('rotation', 0)
        
        # Place in front of table
        if rot == 0:
            chair['x'] = table['x'] + table['width']//2 - chair['width']//2
            chair['y'] = table['y'] + table['height'] + 10
        elif rot == 180:
            chair['x'] = table['x'] + table['width']//2 - chair['width']//2
            chair['y'] = table['y'] - chair['height'] - 10
        elif rot == 90:
            chair['x'] = table['x'] - chair['width'] - 10
            chair['y'] = table['y'] + table['height']//2 - chair['height']//2
        else:  # 270
            chair['x'] = table['x'] + table['width'] + 10
            chair['y'] = table['y'] + table['height']//2 - chair['height']//2
        
        chair['rotation'] = rot
        return chair
    
    def _place_sofa(self, sofa, layout):
        """Place sofa against available wall"""
        # Find empty wall
        walls = ['top', 'bottom', 'left', 'right']
        chosen_wall = random.choice(walls)
        
        if chosen_wall in ['top', 'bottom']:
            sofa['x'] = random.randrange(0, max(1, self.room['width'] - sofa['width']), 
                                         self.grid_step)
            sofa['y'] = 10 if chosen_wall == 'top' else self.room['height'] - sofa['height'] - 10
        else:
            sofa['x'] = 10 if chosen_wall == 'left' else self.room['width'] - sofa['width'] - 10
            sofa['y'] = random.randrange(0, max(1, self.room['height'] - sofa['height']), 
                                         self.grid_step)
        
        sofa['rotation'] = 0
        return sofa
    
    def _place_generic(self, item, layout):
        """Generic random placement"""
        item['x'] = random.randrange(0, max(1, self.room['width'] - item['width']), 
                                     self.grid_step)
        item['y'] = random.randrange(0, max(1, self.room['height'] - item['height']), 
                                     self.grid_step)
        item['rotation'] = random.choice([0, 90, 180, 270])
        return item
    
    def _count_violations(self, layout):
        """Count violations using validator"""
        validator = LayoutValidator(layout)
        return validator.validate()
    
    def _is_valid(self, layout):
        """Check if layout is DIN compliant"""
        if layout is None:
            return False
        violations = self._count_violations(layout)
        return len(violations) == 0