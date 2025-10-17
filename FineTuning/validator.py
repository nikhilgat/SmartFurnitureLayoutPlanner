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

        violations.extend(self._check_bed_clearances()) 
        
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
        # TODO: Implement clearance checking
        return violations
    
    def _check_turning_space(self):
        """At least one 150x150cm turning space"""
        violations = []
        # TODO: Implement turning space check
        return violations
    
    def _check_door_swing(self):
        """90cm door swing arc must be clear"""
        violations = []
        # TODO: Implement door swing check
        return violations
    
    def _check_emergency_path(self):
        """Clear path from bed to door"""
        violations = []
        # TODO: Implement emergency path check
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
        
    def _check_bed_clearances(self):
        """Bed needs 150cm + 120cm on long sides"""
        violations = []
        
        # Find bed
        bed = next((f for f in self.furniture if 'bed' in f['name'].lower()), None)
        if not bed:
            return violations
        
        # Get bed dimensions (considering rotation)
        bed_poly = self._get_furniture_polygon(bed)
        rot = bed.get('rotation', 0)
        
        # Determine long sides based on rotation
        if rot in [0, 180]:
            long_side_length = bed['height']
            # Left and right sides are long
            sides = [
                ('left', box(bed['x'] - 150, bed['y'], bed['x'], bed['y'] + bed['height'])),
                ('right', box(bed['x'] + bed['width'], bed['y'], 
                            bed['x'] + bed['width'] + 150, bed['y'] + bed['height']))
            ]
        else:  # 90 or 270
            long_side_length = bed['width']
            # Top and bottom are long
            sides = [
                ('top', box(bed['x'], bed['y'] - 150, bed['x'] + bed['width'], bed['y'])),
                ('bottom', box(bed['x'], bed['y'] + bed['height'], 
                            bed['x'] + bed['width'], bed['y'] + bed['height'] + 150))
            ]
        
        # Check each side clearance
        side_clearances = []
        room_bounds = box(0, 0, self.room['width'], self.room['height'])
        
        for side_name, clearance_zone in sides:
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
                side_clearances.append(side_name)
        
        # Need at least one side with 150cm clear
        if len(side_clearances) < 1:
            violations.append(
                "Bed clearance: No long side has 150cm clearance available"
            )
        
        return violations