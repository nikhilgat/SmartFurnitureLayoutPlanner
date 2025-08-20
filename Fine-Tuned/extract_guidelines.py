from typing import Dict

def extract_design_guidelines_text(constraints: Dict) -> str:
    guidelines = constraints.get("interior_design_guidelines", {})
    if not guidelines:
        return ""
    
    lines = ["\nADDITIONAL DESIGN GUIDELINES TO FOLLOW:"]
    
    # Functional Costs
    if "functional_costs" in guidelines:
        lines.append("Functional Guidelines:")
        for k, v in guidelines["functional_costs"].items():
            lines.append(f"- {k.replace('_', ' ').capitalize()}: {v}")
    
    # Visual Costs
    if "visual_costs" in guidelines:
        lines.append("\nVisual Composition Rules:")
        for k, v in guidelines["visual_costs"].items():
            lines.append(f"- {k.replace('_', ' ').capitalize()}: {v}")
    
    # Grouping Rules
    if "grouping_rules" in guidelines:
        lines.append("\nGrouping and Interaction Rules:")
        for group, details in guidelines["grouping_rules"].items():
            lines.append(f"- {group.replace('_', ' ').capitalize()}:")
            for k, v in details.items():
                lines.append(f"    • {k.replace('_', ' ').capitalize()}: {v}")
    
    # Fixed Item Support
    if "fixed_item_support" in guidelines:
        lines.append("\nUser Preferences:")
        for k, v in guidelines["fixed_item_support"].items():
            lines.append(f"- {k.replace('_', ' ').capitalize()}: {v}")
    
    return "\n".join(lines)
