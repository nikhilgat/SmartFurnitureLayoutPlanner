import numpy as np
from shapely.geometry import box, Polygon, Point
from shapely.affinity import rotate as shapely_rotate


class LayoutValidator:
    """Validates bedroom layouts against DIN 18040-2 (R)"""
    
    def __init__(self, layout):
        self.room = layout['room']
        self.furniture = layout['furniture']
        self.openings = layout['openings']
        
    def validate(self):
            """Main validation - returns violations list"""
            violations = []
            
            # 1. Check overlaps
            violations.extend(self._check_overlaps())
            
            # 2. Check clearances
            violations.extend(self._check_clearances())
            
            # 3. Check turning space
            violations.extend(self._check_turning_space())
            
            # 4. Check door swing
            violations.extend(self._check_door_swing())
            
            # 5. Check emergency path
            violations.extend(self._check_emergency_path())

            # 6. Check bed clearances
            violations.extend(self._check_bed_clearances())
            
            # 7. Window validations
            violations.extend(self._check_window_clearances())
            violations.extend(self._check_window_reachability())
            violations.extend(self._check_window_access_path())
            
            # 8. Furniture heights (NEW - Priority 1)
            violations.extend(self._check_furniture_heights())
            
            # 9. Reach heights (NEW - Priority 1)
            violations.extend(self._check_reach_heights())
            
            # 10. Door width (NEW - Priority 1)
            violations.extend(self._check_door_width())
            
            # 11. Room size check
            violations.extend(self._check_room_size())
            
            # 12. Sensory and cognitive checks (SUGGESTIONs)
            violations.extend(self._check_sensory_cognitive())
            
            # 13. Door details (additional)
            violations.extend(self._check_door_details())
            
            # 14. Protrusion and flexibility SUGGESTIONs
            violations.extend(self._check_additional_requirements())
            
            return violations
    
    def _get_furniture_polygon(self, item):
        """Convert furniture to Shapely polygon"""
        x, y, w, h = item['x'], item['y'], item['width'], item['height']
        rect = box(x, y, x + w, y + h)
        
        # Apply rotation if exists
        if item.get('rotation', 0) != 0:
            center = (x + w/2, y + h/2)
            rect = shapely_rotate(rect, item['rotation'], origin=center)
        
        return rect
    
    def _check_overlaps(self):
        """No furniture can overlap"""
        violations = []
        polygons = [(f['name'], self._get_furniture_polygon(f)) 
                    for f in self.furniture]
        
        for i, (name1, poly1) in enumerate(polygons):
            for name2, poly2 in polygons[i+1:]:
                if poly1.intersects(poly2):
                    violations.append(f"Overlap: {name1} overlaps {name2}")
        
        return violations
    
    def _check_clearances(self):
        """150cm clearance in front of all furniture"""
        violations = []
        CLEARANCE = 150  # cm
        
        for item in self.furniture:
            clearance_zone = self._get_clearance_zone(item, CLEARANCE)
            
            # Check if clearance overlaps with other furniture
            for other in self.furniture:
                if other['name'] == item['name']:
                    continue
                    
                other_poly = self._get_furniture_polygon(other)
                if clearance_zone.intersects(other_poly):
                    violations.append(
                        f"Clearance violation: {item['name']} needs 150cm "
                        f"clearance but blocked by {other['name']}"
                    )
            
            # Check if clearance is within room bounds
            room_bounds = box(0, 0, self.room['width'], self.room['height'])
            if not room_bounds.contains(clearance_zone):
                violations.append(
                    f"Clearance violation: {item['name']} clearance "
                    f"extends outside room"
                )
        
        return violations

    def _get_clearance_zone(self, item, clearance):
        """Get the 150cm clearance zone in front of furniture"""
        x, y = item['x'], item['y']
        w, h = item['width'], item['height']
        rot = item.get('rotation', 0)
        
        # Default: clearance in front (below furniture in your coordinate system)
        # Adjust based on rotation
        if rot == 0:
            # Front is bottom
            clear_box = box(x, y + h, x + w, y + h + clearance)
        elif rot == 90:
            # Front is left
            clear_box = box(x - clearance, y, x, y + h)
        elif rot == 180:
            # Front is top
            clear_box = box(x, y - clearance, x + w, y)
        elif rot == 270:
            # Front is right
            clear_box = box(x + w, y, x + w + clearance, y + h)
        else:
            # Handle arbitrary rotations
            clear_box = box(x, y + h, x + w, y + h + clearance)
        
        return clear_box
    
    def _check_turning_space(self):
        """At least one 150x150cm turning space exists"""
        violations = []
        TURNING_SIZE = 150  # cm
        
        # Create grid of potential turning spaces
        room_w, room_h = self.room['width'], self.room['height']
        step = 50  # Check every 50cm
        
        found_valid = False
        
        for x in range(0, room_w - TURNING_SIZE + 1, step):
            for y in range(0, room_h - TURNING_SIZE + 1, step):
                turning_zone = box(x, y, x + TURNING_SIZE, y + TURNING_SIZE)
                
                # Check if clear of all furniture
                is_clear = True
                for item in self.furniture:
                    item_poly = self._get_furniture_polygon(item)
                    if turning_zone.intersects(item_poly):
                        is_clear = False
                        break
                
                if is_clear:
                    found_valid = True
                    break
            
            if found_valid:
                break
        
        if not found_valid:
            violations.append(
                "No 150×150cm turning space available - "
                "wheelchair cannot turn in room"
            )
        
        return violations
    
    def _check_door_swing(self):
        """90cm door swing arc must be clear"""
        violations = []
        SWING_RADIUS = 90  # cm
        
        for opening in self.openings:
            if opening['type'] != 'door':
                continue
            
            # Get door position and create swing arc
            swing_zone = self._get_door_swing_zone(opening, SWING_RADIUS)
            
            # Check if any furniture blocks swing
            for item in self.furniture:
                item_poly = self._get_furniture_polygon(item)
                if swing_zone.intersects(item_poly):
                    violations.append(
                        f"Door swing blocked: {item['name']} blocks "
                        f"90cm door swing arc"
                    )
    
        return violations

    def _get_door_swing_zone(self, door, radius):
        """Create door swing arc polygon"""
        wall = door['wall']
        pos = door['position']
        size = door['size']
        
        # Calculate swing arc based on wall
        if wall == 'top':
            # Door on top wall, swings into room (downward)
            center = (pos + size/2, 0)
            # Create arc sector
            swing = Point(center).buffer(radius)
            # Clip to room (swing into room only)
            room_half = box(0, 0, self.room['width'], radius)
            swing = swing.intersection(room_half)
        elif wall == 'bottom':
            center = (pos + size/2, self.room['height'])
            swing = Point(center).buffer(radius)
            room_half = box(0, self.room['height'] - radius, 
                        self.room['width'], self.room['height'])
            swing = swing.intersection(room_half)
        elif wall == 'left':
            center = (0, pos + size/2)
            swing = Point(center).buffer(radius)
            room_half = box(0, 0, radius, self.room['height'])
            swing = swing.intersection(room_half)
        else:  # right
            center = (self.room['width'], pos + size/2)
            swing = Point(center).buffer(radius)
            room_half = box(self.room['width'] - radius, 0, 
                        self.room['width'], self.room['height'])
            swing = swing.intersection(room_half)
        
        return swing
    
    def _check_emergency_path(self):
        """Clear path from bed to door exists"""
        violations = []
        
        # Find bed
        bed = next((f for f in self.furniture if 'bed' in f['name'].lower()), None)
        if not bed:
            return violations  # No bed in layout
        
        # Find door
        door = next((o for o in self.openings if o['type'] == 'door'), None)
        if not door:
            return violations  # No door
        
        # Get bed center
        bed_center = (
            bed['x'] + bed['width']/2,
            bed['y'] + bed['height']/2
        )
        
        # Get door center
        door_pos = self._get_door_center(door)
        
        # Create path line with 90cm width
        from shapely.geometry import LineString
        path_line = LineString([bed_center, door_pos])
        path_zone = path_line.buffer(45)  # 90cm width = 45cm each side
        
        # Check if path blocked by furniture (except bed)
        for item in self.furniture:
            if 'bed' in item['name'].lower():
                continue
            
            item_poly = self._get_furniture_polygon(item)
            if path_zone.intersects(item_poly):
                violations.append(
                    f"Emergency path blocked: {item['name']} blocks "
                    f"direct path from bed to door"
                )
        
        return violations

    def _get_door_center(self, door):
        """Get door center coordinates"""
        wall = door['wall']
        pos = door['position']
        size = door['size']
        
        if wall == 'top':
            return (pos + size/2, 0)
        elif wall == 'bottom':
            return (pos + size/2, self.room['height'])
        elif wall == 'left':
            return (0, pos + size/2)
        else:  # right
            return (self.room['width'], pos + size/2)
        
    def _check_window_clearances(self):
        """Windows need 150cm clearance for operation (with smart exceptions)"""
        violations = []
        CLEARANCE = 150  # cm
        
        for opening in self.openings:
            if opening['type'] != 'window':
                continue
            
            # Get window zone
            window_zone = self._get_window_operation_zone(opening, CLEARANCE)
            sill_height = int(opening.get('heightFromGround', 90))
            
            # Check if furniture blocks window operation
            for item in self.furniture:
                item_poly = self._get_furniture_polygon(item)
                furniture_z_height = int(item.get('zHeight', 100))
                
                if window_zone.intersects(item_poly):
                    # SMART RULE 1: Allow low furniture under high windows
                    if furniture_z_height < sill_height - 15:  # 15cm safety margin
                        continue  # No violation - furniture fits under window
                    
                    # SMART RULE 2: Allow mobile/study furniture (natural light priority)
                    if any(keyword in item['name'].lower() for keyword in ['study', 'chair', 'bedside']):
                        violations.append(
                            f"INFO: {item['name']} near window - ensure operational access maintained"
                        )
                    else:
                        # Hard violation for tall/large furniture
                        violations.append(
                            f"Window operation blocked: {item['name']} (height {furniture_z_height}cm) "
                            f"obstructs window access"
                        )
        
        return violations
    
    def _get_window_operation_zone(self, window, clearance):
        """Get the operation zone in front of a window"""
        wall = window['wall']
        pos = window['position']
        size = window['size']
        
        # Create clearance zone based on wall position
        if wall == 'top':
            # Window on top wall, clearance extends downward into room
            clear_box = box(pos, 0, pos + size, clearance)
        elif wall == 'bottom':
            # Window on bottom wall, clearance extends upward
            clear_box = box(pos, self.room['height'] - clearance, 
                           pos + size, self.room['height'])
        elif wall == 'left':
            # Window on left wall, clearance extends rightward
            clear_box = box(0, pos, clearance, pos + size)
        else:  # right
            # Window on right wall, clearance extends leftward
            clear_box = box(self.room['width'] - clearance, pos, 
                           self.room['width'], pos + size)
        
        return clear_box
    
    def _check_window_reachability(self):
        """Windows should be reachable for wheelchair users"""
        violations = []
        
        # Updated DIN 18040-2 recommendations for wheelchair accessibility
        MIN_SILL_HEIGHT = 60   # cm - minimum for visibility while seated (but practically lower possible)
        MAX_SILL_HEIGHT = 60  # cm - maximum for views while seated
        MAX_HANDLE_HEIGHT = 105  # cm - maximum reach height for handles
        
        for opening in self.openings:
            if opening['type'] != 'window':
                continue
            
            sill_height = int(opening.get('heightFromGround', 90))
            opening_height = int(opening.get('openingHeight', 100))
            handle_height = sill_height + opening_height  # Approximate handle at top
            
            # Check sill height
            if sill_height < MIN_SILL_HEIGHT:
                violations.append(
                    f"Window sill too low: {sill_height}cm "
                    f"(should be ≥{MIN_SILL_HEIGHT}cm for seated visibility)"
                )
            elif sill_height > MAX_SILL_HEIGHT:
                violations.append(
                    f"Window sill too high: {sill_height}cm "
                    f"(should be ≤{MAX_SILL_HEIGHT}cm for seated views)"
                )
            
            # Check handle reachability
            if handle_height > MAX_HANDLE_HEIGHT:
                violations.append(
                    f"Window handle unreachable: {handle_height}cm "
                    f"(should be ≤{MAX_HANDLE_HEIGHT}cm for wheelchair users)"
                )
        
        return violations
    
    def _check_window_access_path(self):
        """Ensure there's a clear path to reach windows"""
        violations = []
        PATH_WIDTH = 90  # cm - minimum width for wheelchair access
        
        for opening in self.openings:
            if opening['type'] != 'window':
                continue
            
            # Get window center point
            window_center = self._get_window_center(opening)
            
            # Check if there's at least 90cm of clear space to approach window
            approach_zone = self._get_window_approach_zone(opening, PATH_WIDTH)
            
            # Check if approach zone is blocked
            blocked = False
            blocking_furniture = []
            
            for item in self.furniture:
                # Skip small/mobile furniture
                if any(keyword in item['name'].lower() for keyword in ['chair', 'bedside']):
                    continue
                
                item_poly = self._get_furniture_polygon(item)
                if approach_zone.intersects(item_poly):
                    blocked = True
                    blocking_furniture.append(item['name'])
            
            if blocked:
                violations.append(
                    f"Window access path blocked: Cannot reach window due to "
                    f"{', '.join(blocking_furniture)} (need {PATH_WIDTH}cm clear approach)"
                )
        
        return violations
    
    def _get_window_center(self, window):
        """Get window center coordinates"""
        wall = window['wall']
        pos = window['position']
        size = window['size']
        
        if wall == 'top':
            return (pos + size/2, 0)
        elif wall == 'bottom':
            return (pos + size/2, self.room['height'])
        elif wall == 'left':
            return (0, pos + size/2)
        else:  # right
            return (self.room['width'], pos + size/2)
    
    def _get_window_approach_zone(self, window, width):
        """Get the approach zone to reach a window (90cm wide path)"""
        wall = window['wall']
        pos = window['position']
        size = window['size']
        half_width = width / 2
        
        # Create approach zone centered on window, extending into room
        if wall == 'top':
            center_pos = pos + size/2
            approach_box = box(center_pos - half_width, 0, 
                              center_pos + half_width, width)
        elif wall == 'bottom':
            center_pos = pos + size/2
            approach_box = box(center_pos - half_width, self.room['height'] - width,
                              center_pos + half_width, self.room['height'])
        elif wall == 'left':
            center_pos = pos + size/2
            approach_box = box(0, center_pos - half_width,
                              width, center_pos + half_width)
        else:  # right
            center_pos = pos + size/2
            approach_box = box(self.room['width'] - width, center_pos - half_width,
                              self.room['width'], center_pos + half_width)
        
        return approach_box
    
    def _check_bed_clearances(self):
        """Bed needs 150cm on one long side and 120cm on the other"""
        violations = []
        
        # Find bed
        bed = next((f for f in self.furniture if 'bed' in f['name'].lower()), None)
        if not bed:
            return violations
        
        # Check bed height
        bed_height = int(bed.get('zHeight', 55))
        if not (45 <= bed_height <= 52):
            violations.append(
                f"Bed height violation: {bed_height}cm "
                f"(should be between 45-52cm for easy transfers)"
            )
        
        # SUGGESTION for under-bed clearance (assuming no data, add info)
        violations.append(
            "INFO: Ensure under-bed clearance ≥67cm for wheelchair footrests in care scenarios"
        )
        
        # Get bed dimensions (considering rotation)
        bed_poly = self._get_furniture_polygon(bed)
        rot = bed.get('rotation', 0)
        
        # Determine long sides based on rotation
        if rot in [0, 180]:
            long_side_length = bed['height']
            # Left and right sides are long
            sides_150 = {
                'left': box(bed['x'] - 150, bed['y'], bed['x'], bed['y'] + bed['height']),
                'right': box(bed['x'] + bed['width'], bed['y'], 
                            bed['x'] + bed['width'] + 150, bed['y'] + bed['height'])
            }
            sides_120 = {
                'left': box(bed['x'] - 120, bed['y'], bed['x'], bed['y'] + bed['height']),
                'right': box(bed['x'] + bed['width'], bed['y'], 
                            bed['x'] + bed['width'] + 120, bed['y'] + bed['height'])
            }
        else:  # 90 or 270
            long_side_length = bed['width']
            # Top and bottom are long
            sides_150 = {
                'top': box(bed['x'], bed['y'] - 150, bed['x'] + bed['width'], bed['y']),
                'bottom': box(bed['x'], bed['y'] + bed['height'], 
                            bed['x'] + bed['width'], bed['y'] + bed['height'] + 150)
            }
            sides_120 = {
                'top': box(bed['x'], bed['y'] - 120, bed['x'] + bed['width'], bed['y']),
                'bottom': box(bed['x'], bed['y'] + bed['height'], 
                            bed['x'] + bed['width'], bed['y'] + bed['height'] + 120)
            }
        
        # Check each side clearance
        room_bounds = box(0, 0, self.room['width'], self.room['height'])
        
        clear_150_sides = []
        for side_name, clearance_zone in sides_150.items():
            # Check if in room bounds
            if not room_bounds.contains(clearance_zone):
                continue
            
            # Check if blocked
            blocked = False
            for item in self.furniture:
                if 'bed' in item['name'].lower():
                    continue
                item_poly = self._get_furniture_polygon(item)
                if clearance_zone.intersects(item_poly):
                    blocked = True
                    break
            
            if not blocked:
                clear_150_sides.append(side_name)
        
        # Need at least one side with 150cm clear
        has_150 = len(clear_150_sides) >= 1
        has_120_on_other = False
        
        if has_150:
            for clear_side in clear_150_sides:
                other_sides = [s for s in sides_150 if s != clear_side]
                for other_side in other_sides:
                    clearance_zone_120 = sides_120[other_side]
                    if not room_bounds.contains(clearance_zone_120):
                        continue
                    blocked_120 = False
                    for item in self.furniture:
                        if 'bed' in item['name'].lower():
                            continue
                        item_poly = self._get_furniture_polygon(item)
                        if clearance_zone_120.intersects(item_poly):
                            blocked_120 = True
                            break
                    if not blocked_120:
                        has_120_on_other = True
                        break
                if has_120_on_other:
                    break
        
        if not has_150:
            violations.append(
                "Bed clearance: No long side has 150cm clearance available"
            )
        if has_150 and not has_120_on_other:
            violations.append(
                "Bed clearance: No other long side has at least 120cm clearance available"
            )
        
        return violations
    
    def _check_furniture_heights(self):
        """Check furniture heights comply with DIN 18040-2 accessibility standards"""
        violations = []
        
        # DIN 18040-2 height standards (cm)
        HEIGHT_STANDARDS = {
            # Work surfaces (tables, desks)
            'work_surface': {
                'furniture_types': ['study table', 'desk', 'table', 'dining table'],
                'min_height': 75,
                'max_height': 85,
                'reason': 'wheelchair accessibility'
            },
            # Seating
            'seating': {
                'furniture_types': ['chair', 'sofa', 'bench'],
                'min_height': 46,
                'max_height': 50,
                'reason': 'easy transfer for wheelchair users'
            },
            # Storage (reachable)
            'storage_low': {
                'furniture_types': ['bedside table', 'nightstand', 'side table'],
                'min_height': 40,
                'max_height': 60,
                'reason': 'within reach from bed'
            },
            # High storage should not exceed reach height
            'storage_high': {
                'furniture_types': ['wardrobe', 'closet', 'cabinet', 'shelf'],
                'min_height': 0,
                'max_height': 140,  # Top shelf max reach
                'reason': 'wheelchair reach limit',
                'check_type': 'SUGGESTION'  # Just a SUGGESTION, not hard violation
            }
        }
        
        for item in self.furniture:
            furniture_z_height = int(item.get('zHeight', 0))
            item_name_lower = item['name'].lower()
            
            # Check against each standard
            for standard_name, standard in HEIGHT_STANDARDS.items():
                # Check if furniture matches this category
                if any(ftype in item_name_lower for ftype in standard['furniture_types']):
                    min_h = standard['min_height']
                    max_h = standard['max_height']
                    reason = standard['reason']
                    is_SUGGESTION = standard.get('check_type') == 'SUGGESTION'
                    
                    # Check height compliance
                    if furniture_z_height < min_h:
                        msg = (f"Height violation: {item['name']} at {furniture_z_height}cm "
                               f"(should be ≥{min_h}cm for {reason})")
                        violations.append(msg if not is_SUGGESTION else f"SUGGESTION: {msg}")
                    
                    elif furniture_z_height > max_h:
                        msg = (f"Height violation: {item['name']} at {furniture_z_height}cm "
                               f"(should be ≤{max_h}cm for {reason})")
                        violations.append(msg if not is_SUGGESTION else f"SUGGESTION: {msg}")
                    
                    break  # Only check first matching category
        
        return violations
    
    def _check_reach_heights(self):
        """Check if bedside table is within reach from bed"""
        violations = []
        
        # Find bed
        bed = next((f for f in self.furniture if 'bed' in f['name'].lower()), None)
        if not bed:
            return violations  # No bed to check
        
        # Find bedside tables/nightstands
        bedside_items = [f for f in self.furniture 
                        if any(keyword in f['name'].lower() 
                              for keyword in ['bedside', 'nightstand', 'side table'])]
        
        if not bedside_items:
            return violations  # No bedside furniture
        
        # DIN 18040-2 reach standards
        MAX_REACH_HORIZONTAL = 60  # cm - max comfortable reach from lying position
        MAX_HEIGHT_DIFF = 15  # cm - height difference between bed and bedside table
        
        bed_height = int(bed.get('zHeight', 55))
        bed_center_x = bed['x'] + bed['width'] / 2
        bed_center_y = bed['y'] + bed['height'] / 2
        
        for bedside in bedside_items:
            bedside_height = int(bedside.get('zHeight', 60))
            bedside_center_x = bedside['x'] + bedside['width'] / 2
            bedside_center_y = bedside['y'] + bedside['height'] / 2
            
            # Calculate distance between bed and bedside table
            distance = ((bed_center_x - bedside_center_x)**2 + 
                       (bed_center_y - bedside_center_y)**2)**0.5
            
            # Check horizontal reach
            if distance > MAX_REACH_HORIZONTAL:
                violations.append(
                    f"Reach violation: {bedside['name']} is {distance:.0f}cm from bed "
                    f"(should be ≤{MAX_REACH_HORIZONTAL}cm for easy reach)"
                )
            
            # Check height compatibility
            height_diff = abs(bed_height - bedside_height)
            if height_diff > MAX_HEIGHT_DIFF:
                violations.append(
                    f"Height reach violation: {bedside['name']} height ({bedside_height}cm) "
                    f"differs from bed ({bed_height}cm) by {height_diff}cm "
                    f"(should be ≤{MAX_HEIGHT_DIFF}cm for easy reach)"
                )
        
        return violations
    
    def _check_door_width(self):
        """Check door width meets minimum 90cm requirement"""
        violations = []
        
        MIN_DOOR_WIDTH = 90  # cm - DIN 18040-2 minimum clear width
        RECOMMENDED_DOOR_WIDTH = 100  # cm - recommended for easier wheelchair passage
        
        for opening in self.openings:
            if opening['type'] != 'door':
                continue
            
            door_width = opening.get('size', 0)
            
            # Check minimum width
            if door_width < MIN_DOOR_WIDTH:
                violations.append(
                    f"Door width violation: Door is {door_width}cm wide "
                    f"(must be ≥{MIN_DOOR_WIDTH}cm for wheelchair access)"
                )
            # Optional: Add info for suboptimal but acceptable widths
            elif door_width < RECOMMENDED_DOOR_WIDTH:
                violations.append(
                    f"INFO: Door is {door_width}cm wide "
                    f"(recommended ≥{RECOMMENDED_DOOR_WIDTH}cm for comfortable wheelchair passage)"
                )
        
        return violations
    
    def _check_room_size(self):
        """Check minimum room size for bedroom"""
        violations = []
        MIN_AREA_SINGLE = 100000  # 10 m² in cm² (assuming single bed)
        
        area = self.room['width'] * self.room['height']
        if area < MIN_AREA_SINGLE:
            violations.append(
                f"Room size violation: Area {area / 10000:.1f}m² "
                f"(should be ≥10m² for single bedroom accessibility)"
            )
        
        return violations
    
    def _check_sensory_cognitive(self):
        """Add SUGGESTIONs for sensory and cognitive requirements"""
        violations = []
        violations.append("SUGGESTION: Ensure luminance contrasts ≥0.4 for visual impairments")
        violations.append("SUGGESTION: Provide tactile guides for orientation")
        violations.append("SUGGESTION: Use glare-free lighting (100-300 lux)")
        violations.append("SUGGESTION: Floors should be slip-resistant (R9 rating)")
        violations.append("SUGGESTION: Include two-sense emergency alarms (visual/audible)")
        
        return violations
    
    def _check_door_details(self):
        violations = []
        MIN_DOOR_HEIGHT = 205  # cm
        MAX_THRESHOLD = 2  # cm
        MAX_FORCE = 30  # N

        for opening in self.openings:
            if opening['type'] != 'door':
                continue

            door_height = opening.get('openingHeight', 0)

            try:
                door_height = int(door_height)
            except (ValueError, TypeError):
                door_height = 0

            if door_height and door_height < MIN_DOOR_HEIGHT:
                violations.append(
                    f"Door height violation: {door_height}cm "
                    f"(should be ≥{MIN_DOOR_HEIGHT}cm)"
                )
            else:
                violations.append("INFO: Ensure door height ≥205cm")

            violations.append("INFO: Ensure door thresholds ≤2cm")
            violations.append("INFO: Ensure door opening force ≤30N")
            violations.append("INFO: Prefer outward-opening doors for safety")

        return violations

    
    def _check_additional_requirements(self):
        """Add SUGGESTIONs for protrusions, flexibility, etc."""
        violations = []
        violations.append("SUGGESTION: Do checks for wall protrusions - ensure ≤15cm above floor")
        violations.append("SUGGESTION: Ensure layout flexibility for care aids (removable elements)")
        violations.append("INFO: Consider integration with corridors (≥120cm wide paths)")
        
        return violations