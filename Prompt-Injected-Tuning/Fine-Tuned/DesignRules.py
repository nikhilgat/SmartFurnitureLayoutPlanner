"""
Professional Care Home Design Rules
Combines DIN 18040-2 with professional interior design principles
Based on actual DIN guideline diagrams and professional care home standards
"""

import json
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field

@dataclass
class FurnitureRelationship:
    """Defines how furniture items should be spatially related"""
    primary: str
    companion: List[str]
    relationship_type: str  # "adjacent", "nearby", "zone_member"
    min_distance: int  # cm
    max_distance: int  # cm
    preferred_distance: int  # cm
    description: str

@dataclass
class DesignZone:
    """Functional zone in a care home bedroom"""
    name: str
    furniture_types: List[str]
    min_area: int  # cm²
    preferred_location: str  # "window", "corner", "wall", "center", "door_opposite"
    clearance_required: int  # cm
    priority: int  # 1-5, 5 being highest
    description: str

@dataclass
class DINCompliance:
    """DIN 18040-2 specific requirements"""
    requirement_name: str
    measurement: int  # cm
    mandatory: bool
    wheelchair_specific: bool
    reference: str
    description: str

class ProfessionalCareHomeRules:
    """
    Complete rule system for professional care home bedroom layouts
    Extracted from DIN 18040-2 Section 5.4 (Schlafräume - Bedrooms)
    """
    
    def __init__(self):
        self.din_requirements = self._init_din_requirements()
        self.furniture_relationships = self._init_furniture_relationships()
        self.design_zones = self._init_design_zones()
        self.clearance_matrix = self._init_clearance_matrix()
        self.design_principles = self._init_design_principles()
    
    def _init_din_requirements(self) -> Dict[str, DINCompliance]:
        """
        DIN 18040-2 Section 5.4 Requirements
        Based on official standard and guideline diagrams
        """
        return {
            # Turning Spaces (Section 5.4)
            "turning_space_standard": DINCompliance(
                requirement_name="Minimum turning space with walker",
                measurement=120,  # 120x120 cm
                mandatory=True,
                wheelchair_specific=False,
                reference="DIN 18040-2, 5.4",
                description="Minimum 120x120cm clear space for turning with walking aids"
            ),
            "turning_space_wheelchair": DINCompliance(
                requirement_name="Wheelchair turning circle",
                measurement=150,  # 150x150 cm
                mandatory=True,
                wheelchair_specific=True,
                reference="DIN 18040-2, 5.4 (R)",
                description="Minimum 150x150cm clear turning space for wheelchair users"
            ),
            
            # Bed Clearances (Section 5.4)
            "bed_clearance_primary_standard": DINCompliance(
                requirement_name="Bed access - primary side (standard)",
                measurement=120,
                mandatory=True,
                wheelchair_specific=False,
                reference="DIN 18040-2, 5.4",
                description="120cm clearance along one long side of bed"
            ),
            "bed_clearance_secondary_standard": DINCompliance(
                requirement_name="Bed access - secondary side (standard)",
                measurement=90,
                mandatory=True,
                wheelchair_specific=False,
                reference="DIN 18040-2, 5.4",
                description="90cm clearance along other long side of bed"
            ),
            "bed_clearance_primary_wheelchair": DINCompliance(
                requirement_name="Bed access - primary side (wheelchair)",
                measurement=150,
                mandatory=True,
                wheelchair_specific=True,
                reference="DIN 18040-2, 5.4 (R)",
                description="150cm clearance on main access side for wheelchair transfer"
            ),
            "bed_clearance_secondary_wheelchair": DINCompliance(
                requirement_name="Bed access - secondary side (wheelchair)",
                measurement=120,
                mandatory=True,
                wheelchair_specific=True,
                reference="DIN 18040-2, 5.4 (R)",
                description="120cm clearance on opposite side"
            ),
            
            # General Furniture Clearances (Section 5.4)
            "furniture_clearance_standard": DINCompliance(
                requirement_name="General furniture front clearance",
                measurement=90,
                mandatory=True,
                wheelchair_specific=False,
                reference="DIN 18040-2, 5.4",
                description="90cm minimum in front of wardrobes, desks, etc."
            ),
            "furniture_clearance_wheelchair": DINCompliance(
                requirement_name="Furniture clearance for wheelchair",
                measurement=150,
                mandatory=True,
                wheelchair_specific=True,
                reference="DIN 18040-2, 5.4 (R)",
                description="150cm clearance for wheelchair approach to furniture"
            ),
            
            # Pathway Widths (Section 4.3.2)
            "pathway_minimum": DINCompliance(
                requirement_name="Minimum pathway width",
                measurement=120,
                mandatory=True,
                wheelchair_specific=False,
                reference="DIN 18040-2, 4.3.2",
                description="120cm minimum for movement corridors"
            ),
            "pathway_preferred": DINCompliance(
                requirement_name="Preferred pathway width",
                measurement=150,
                mandatory=False,
                wheelchair_specific=True,
                reference="DIN 18040-2, 4.3.2",
                description="150cm preferred for comfortable wheelchair movement"
            ),
            
            # Door Clearances (Section 4.3.3)
            "door_clearance": DINCompliance(
                requirement_name="Door maneuvering space",
                measurement=150,
                mandatory=True,
                wheelchair_specific=True,
                reference="DIN 18040-2, 4.3.3",
                description="150cm clearance at doors for wheelchair maneuvering"
            ),
        }
    
    def _init_furniture_relationships(self) -> List[FurnitureRelationship]:
        """
        Professional furniture relationships for care home bedrooms
        Based on ergonomics, functionality, and care requirements
        """
        return [
            # Bed Relationships
            FurnitureRelationship(
                primary="Bed",
                companion=["Bedside Table", "Nightstand"],
                relationship_type="adjacent",
                min_distance=20,
                max_distance=60,
                preferred_distance=30,
                description="Bedside table must be within arm's reach (30-50cm) from bed"
            ),
            FurnitureRelationship(
                primary="Bed",
                companion=["Wardrobe", "Closet", "Dresser"],
                relationship_type="nearby",
                min_distance=150,
                max_distance=400,
                preferred_distance=200,
                description="Wardrobe near bed but with proper circulation clearance"
            ),
            
            # Desk/Work Area Relationships
            FurnitureRelationship(
                primary="Study Table",
                companion=["Study Chair", "Desk Chair", "Chair"],
                relationship_type="adjacent",
                min_distance=5,
                max_distance=20,
                preferred_distance=10,
                description="Chair positioned directly at desk for functional use"
            ),
            FurnitureRelationship(
                primary="Desk",
                companion=["Study Chair", "Desk Chair", "Chair"],
                relationship_type="adjacent",
                min_distance=5,
                max_distance=20,
                preferred_distance=10,
                description="Chair placement for desk ergonomics and accessibility"
            ),
            
            # Seating Area Relationships
            FurnitureRelationship(
                primary="Sofa",
                companion=["Coffee Table", "Side Table"],
                relationship_type="adjacent",
                min_distance=40,
                max_distance=80,
                preferred_distance=50,
                description="Table within comfortable reach from seated position"
            ),
            FurnitureRelationship(
                primary="Armchair",
                companion=["Side Table", "Lamp"],
                relationship_type="adjacent",
                min_distance=30,
                max_distance=70,
                preferred_distance=40,
                description="Side table for personal items within reach"
            ),
        ]
    
    def _init_design_zones(self) -> List[DesignZone]:
        """
        Functional zones in care home bedrooms
        Based on DIN diagrams showing proper zone organization
        """
        return [
            DesignZone(
                name="Sleep Zone",
                furniture_types=["Bed", "Bedside Table", "Nightstand"],
                min_area=300 * 250,  # 7.5 m²
                preferred_location="corner",
                clearance_required=150,
                priority=5,
                description="Primary rest area - bed with bilateral or unilateral access per DIN"
            ),
            DesignZone(
                name="Storage Zone",
                furniture_types=["Wardrobe", "Closet", "Dresser"],
                min_area=200 * 150,  # 3 m²
                preferred_location="wall",
                clearance_required=150,
                priority=4,
                description="Clothing and personal item storage with full wheelchair access"
            ),
            DesignZone(
                name="Work/Activity Zone",
                furniture_types=["Desk", "Study Table", "Study Chair", "Chair"],
                min_area=180 * 150,  # 2.7 m²
                preferred_location="window",
                clearance_required=120,
                priority=3,
                description="Workspace with natural light access when available"
            ),
            DesignZone(
                name="Seating Zone",
                furniture_types=["Sofa", "Armchair", "Coffee Table", "Side Table"],
                min_area=250 * 200,  # 5 m²
                preferred_location="door_opposite",
                clearance_required=120,
                priority=3,
                description="Comfortable seating area for relaxation and visitors"
            ),
            DesignZone(
                name="Circulation Zone",
                furniture_types=[],  # No furniture
                min_area=150 * 150,  # 2.25 m² minimum
                preferred_location="center",
                clearance_required=150,
                priority=5,
                description="Central clear area for wheelchair turning (150x150cm per DIN)"
            ),
        ]
    
    def _init_clearance_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Clearance requirements between furniture types
        Based on DIN 18040-2 and professional ergonomics
        """
        return {
            "Bed": {
                "primary_side": 150,  # Main access for wheelchair
                "secondary_side": 120,  # Secondary access
                "foot": 90,  # Less critical
                "head": 90,
            },
            "Wardrobe": {
                "front": 150,  # Full wheelchair access
                "sides": 20,  # Can be tight
            },
            "Closet": {
                "front": 150,
                "sides": 20,
            },
            "Dresser": {
                "front": 120,
                "sides": 30,
            },
            "Desk": {
                "front": 150,  # Wheelchair approach
                "back": 80,
                "sides": 60,
            },
            "Study Table": {
                "front": 150,
                "back": 80,
                "sides": 60,
            },
            "Sofa": {
                "front": 120,
                "back": 50,
                "sides": 80,
            },
            "Armchair": {
                "front": 100,
                "back": 40,
                "sides": 60,
            },
            "Chair": {
                "all": 60,  # Minimal clearance
            },
            "Bedside Table": {
                "all": 40,  # Small item
            },
        }
    
    def _init_design_principles(self) -> Dict[str, Dict]:
        """
        Professional design principles for care home layouts
        """
        return {
            "accessibility": {
                "priority": 10,
                "rules": [
                    "Wheelchair turning space (150x150cm) must always be available",
                    "Primary pathways must be minimum 120cm, preferably 150cm",
                    "Bed must have 150cm clearance on primary access side",
                    "All furniture must have appropriate front clearance",
                    "Door swing areas must remain clear",
                ]
            },
            "functionality": {
                "priority": 9,
                "rules": [
                    "Related furniture must be grouped (bed+nightstand, desk+chair)",
                    "Frequently used items should be easily accessible",
                    "Work areas should have natural light when possible",
                    "Storage should be distributed logically",
                ]
            },
            "ergonomics": {
                "priority": 8,
                "rules": [
                    "Bedside table within 30-50cm of bed",
                    "Desk chair positioned for comfortable use",
                    "Seating areas allow easy reach to tables",
                    "Clearances accommodate mobility aids",
                ]
            },
            "safety": {
                "priority": 10,
                "rules": [
                    "Clear pathways without obstacles",
                    "Adequate space for emergency access",
                    "No furniture blocking exits",
                    "Stable furniture placement",
                ]
            },
            "aesthetics": {
                "priority": 6,
                "rules": [
                    "Balanced room layout",
                    "Visual harmony in furniture placement",
                    "Proper use of space",
                    "Consideration of views and natural light",
                ]
            }
        }
    
    def get_clearance_for_furniture(self, furniture_type: str, 
                                   wheelchair: bool = True) -> Dict[str, int]:
        """Get required clearances for a furniture type"""
        base_clearances = self.clearance_matrix.get(furniture_type, {"all": 90})
        
        # Apply wheelchair adjustments
        if wheelchair:
            adjusted = {}
            for direction, value in base_clearances.items():
                # Ensure minimum 120cm for wheelchair access
                adjusted[direction] = max(value, 120)
            return adjusted
        
        return base_clearances
    
    def get_relationships_for_furniture(self, furniture_name: str) -> List[FurnitureRelationship]:
        """Get all relationships where this furniture is primary"""
        return [r for r in self.furniture_relationships if r.primary in furniture_name]
    
    def get_zone_for_furniture(self, furniture_type: str) -> DesignZone:
        """Determine which zone a furniture piece belongs to"""
        for zone in self.design_zones:
            for zone_type in zone.furniture_types:
                if zone_type.lower() in furniture_type.lower():
                    return zone
        # Default to circulation if not found
        return self.design_zones[-1]  # Circulation zone
    
    def export_rules(self, filename: str = "professional_design_rules.json"):
        """Export all rules to JSON for reference"""
        export_data = {
            "din_requirements": {k: v.__dict__ for k, v in self.din_requirements.items()},
            "furniture_relationships": [r.__dict__ for r in self.furniture_relationships],
            "design_zones": [z.__dict__ for z in self.design_zones],
            "clearance_matrix": self.clearance_matrix,
            "design_principles": self.design_principles,
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Design rules exported to {filename}")
        return export_data


# Test and export
if __name__ == "__main__":
    print("="*60)
    print("PROFESSIONAL CARE HOME DESIGN RULES")
    print("Based on DIN 18040-2 Section 5.4")
    print("="*60)
    
    rules = ProfessionalCareHomeRules()
    
    print("\n✓ DIN Requirements loaded:", len(rules.din_requirements))
    print("✓ Furniture Relationships defined:", len(rules.furniture_relationships))
    print("✓ Design Zones configured:", len(rules.design_zones))
    print("✓ Design Principles established:", len(rules.design_principles))
    
    # Export for reference
    rules.export_rules()
    
    # Show example relationship
    print("\n" + "="*60)
    print("EXAMPLE FURNITURE RELATIONSHIP:")
    print("="*60)
    bed_relationships = rules.get_relationships_for_furniture("Bed")
    for rel in bed_relationships:
        print(f"\n{rel.primary} → {rel.companion}")
        print(f"  Type: {rel.relationship_type}")
        print(f"  Preferred distance: {rel.preferred_distance}cm")
        print(f"  Reason: {rel.description}")
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETE!")
    print("="*60)
    print("\nNext: Run this file to generate professional_design_rules.json")
    print("Then we'll move to Step 2: Layout Optimizer")