"""
Visualization Generator for Cost Function Analysis
Creates publication-quality figures for thesis

Usage: python visualize_costs.py <original_layout.json> <optimized_layout.json>

Generates:
- Radar chart comparing cost components
- Bar chart of improvements
- Component breakdown visualization
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from costFunction.cost_validator import CostFunctionValidator


def create_radar_chart(original_costs, optimized_costs, output_file='cost_radar.png'):
    """
    Create radar chart comparing cost components
    """
    
    categories = ['Flow\n& Adjacency', 'Zoning', 'Environmental', 
                  'Clearance\n& Ergonomics', 'Aesthetics']
    
    # Normalize costs to 0-1 scale (lower is better, so invert)
    components = ['C_flow', 'C_zone', 'C_env', 'C_clearance', 'C_vis']
    
    # Get max values for normalization
    max_vals = {}
    for comp in components:
        max_vals[comp] = max(original_costs['components'][comp], 
                            optimized_costs['components'][comp])
        if max_vals[comp] == 0:
            max_vals[comp] = 1  # Avoid division by zero
    
    # Normalize and invert (so higher = better in visualization)
    original_scores = []
    optimized_scores = []
    
    for comp in components:
        orig_norm = 1 - (original_costs['components'][comp] / max_vals[comp])
        opt_norm = 1 - (optimized_costs['components'][comp] / max_vals[comp])
        
        original_scores.append(orig_norm)
        optimized_scores.append(opt_norm)
    
    # Setup radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    original_scores += original_scores[:1]
    optimized_scores += optimized_scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, original_scores, 'o-', linewidth=2, 
            label='Original Layout', color='#e74c3c', markersize=8)
    ax.plot(angles, optimized_scores, 'o-', linewidth=2, 
            label='Optimized Layout', color='#27ae60', markersize=8)
    ax.fill(angles, optimized_scores, alpha=0.25, color='#27ae60')
    
    # Customize chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    # Title
    plt.title('Multi-Objective Cost Function Comparison\n(Higher is Better)', 
              size=14, pad=20, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved radar chart: {output_file}")
    
    plt.close()


def create_improvement_bar_chart(original_costs, optimized_costs, 
                                 output_file='improvement_bars.png'):
    """
    Create bar chart showing improvement percentages
    """
    
    components = ['C_flow', 'C_zone', 'C_env', 'C_clearance', 'C_vis']
    component_names = ['Flow &\nAdjacency', 'Zoning', 'Environmental', 
                      'Clearance &\nErgonomics', 'Aesthetics']
    
    improvements = []
    for comp in components:
        orig = original_costs['components'][comp]
        opt = optimized_costs['components'][comp]
        
        if orig > 0:
            improvement = ((orig - opt) / orig) * 100
        else:
            improvement = 0 if opt == 0 else -100
        
        improvements.append(improvement)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars based on improvement (green=good, red=bad)
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    
    # Create bars
    bars = ax.bar(component_names, improvements, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{improvement:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    # Customize
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Cost Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cost Component', fontsize=12, fontweight='bold')
    ax.set_title('Component-wise Cost Improvement\n(Positive = Better)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', alpha=0.7, edgecolor='black', label='Improved'),
        Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', label='Degraded')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved improvement bar chart: {output_file}")
    
    plt.close()


def create_cost_breakdown(original_costs, optimized_costs, 
                         output_file='cost_breakdown.png'):
    """
    Create stacked bar chart showing weighted cost breakdown
    """
    
    components = ['C_flow', 'C_zone', 'C_env', 'C_clearance', 'C_vis']
    component_names = ['Flow &\nAdjacency', 'Zoning', 'Environmental', 
                      'Clearance &\nErgonomics', 'Aesthetics']
    
    # Get weighted costs
    original_weighted = [original_costs['weighted_components'][c] for c in components]
    optimized_weighted = [optimized_costs['weighted_components'][c] for c in components]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(component_names))
    width = 0.35
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, original_weighted, width, 
                   label='Original', color='#e74c3c', alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, optimized_weighted, width, 
                   label='Optimized', color='#27ae60', alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
    
    # Customize
    ax.set_ylabel('Weighted Cost', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cost Component', fontsize=12, fontweight='bold')
    ax.set_title('Weighted Cost Component Comparison\n(Lower is Better)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(component_names)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved cost breakdown: {output_file}")
    
    plt.close()


def create_total_cost_comparison(original_costs, optimized_costs,
                                 output_file='total_cost.png'):
    """
    Create simple comparison of total costs
    """
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['Original\nLayout', 'Optimized\nLayout']
    costs = [original_costs['total'], optimized_costs['total']]
    colors = ['#e74c3c', '#27ae60']
    
    bars = ax.bar(labels, costs, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=2, width=0.5)
    
    # Add values on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{cost:.1f}',
               ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Calculate improvement
    improvement = ((costs[0] - costs[1]) / costs[0]) * 100
    
    # Add improvement annotation
    ax.annotate('', xy=(1, costs[1]), xytext=(1, costs[0]),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(1.15, (costs[0] + costs[1])/2, f'{improvement:+.1f}%\nimprovement',
            fontsize=12, fontweight='bold', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Customize
    ax.set_ylabel('Total Weighted Cost', fontsize=13, fontweight='bold')
    ax.set_title('Total Cost Comparison\n(Lower is Better)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved total cost comparison: {output_file}")
    
    plt.close()


def generate_all_visualizations(original_file, optimized_file, output_prefix=''):
    """
    Generate all visualization types
    """
    
    print("="*80)
    print(" VISUALIZATION GENERATOR")
    print("="*80)
    
    # Load layouts
    print(f"\n[*] Loading layouts...")
    try:
        with open(original_file, 'r') as f:
            original_layout = json.load(f)
        print(f"  [OK] Original: {original_file}")
    except Exception as e:
        print(f"  [X] Error loading original: {e}")
        return
    
    try:
        with open(optimized_file, 'r') as f:
            optimized_layout = json.load(f)
        print(f"  [OK] Optimized: {optimized_file}")
    except Exception as e:
        print(f"  [X] Error loading optimized: {e}")
        return
    
    # Calculate costs
    print(f"\n[*] Calculating costs...")
    original_validator = CostFunctionValidator(original_layout)
    original_costs = original_validator.calculate_total_cost()
    
    optimized_validator = CostFunctionValidator(optimized_layout)
    optimized_costs = optimized_validator.calculate_total_cost()
    
    print(f"  [OK] Original total cost: {original_costs['total']:.2f}")
    print(f"  [OK] Optimized total cost: {optimized_costs['total']:.2f}")
    
    improvement = ((original_costs['total'] - optimized_costs['total']) / 
                   original_costs['total']) * 100
    print(f"  [OK] Improvement: {improvement:+.2f}%")
    
    # Generate visualizations
    print(f"\n[*] Generating visualizations...")
    
    create_radar_chart(original_costs, optimized_costs, 
                      f'{output_prefix}cost_radar.png')
    
    create_improvement_bar_chart(original_costs, optimized_costs,
                                f'{output_prefix}improvement_bars.png')
    
    create_cost_breakdown(original_costs, optimized_costs,
                         f'{output_prefix}cost_breakdown.png')
    
    create_total_cost_comparison(original_costs, optimized_costs,
                                f'{output_prefix}total_cost.png')

def main():
    """Main entry point"""
    
    original_file = "Input-Layouts/room-layout (11).json"
    optimized_file = "Output-Layouts/room-layout (11)-optimized.json"
    
    generate_all_visualizations(original_file, optimized_file)


if __name__ == "__main__":
    main()
