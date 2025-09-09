# SmartFurnitureLayoutPlanner

**SmartFurnitureLayoutPlanner** explores multiple approaches to automatic furniture layout generation and optimization for interior spaces. The repository is organized by approach (traditional heuristics, reinforcement learning, and fine-tuned ML), with shared *inputs*, *constraints*, and *outputs*. Sample room JSONs and a rendered “best” layout image are included to make the pipeline reproducible end-to-end. ([GitHub][1])

> Key goals
>
> * Generate valid, non-overlapping layouts that respect circulation and accessibility constraints
> * Optimize for criteria like **barrier-free access**, clearances, door/window/switch reach, and functional adjacencies
> * Compare **Traditional** (rule/heuristic), **Reinforcement-Learning**, and **Fine-Tuned** ML strategies on the same inputs/constraints

---

## Repository map

```
SmartFurnitureLayoutPlanner/
├─ Traditional/                 # Heuristic/rule-based placement & local search
├─ Reinforcement-Learning/      # RL agent(s) for sequential placement/adjustment
├─ Fine-Tuned/                  # Fine-tuned ML models (e.g., learned scoring)
├─ Input-Layouts/               # Sample room JSONs & minimal test cases
├─ Outputs/                     # Generated layouts & visualizations
├─ constraints/                 # Constraint definitions / parameters
├─ untrack/                     # (Local stuff ignored by git; logs, cache, etc.)
├─ best_barrier_free_layout.png # Example “best” output visualization
├─ README.md
└─ LICENSE
```

> The folder names and top-level files are directly visible in the repo’s landing page. ([GitHub][1])

---

## What each folder contains (and how to use it)

### `Input-Layouts/`

* **Purpose:** Canonical source of input rooms used across all methods for apples-to-apples comparisons.
* **Contents:** JSON files describing the shell (walls, doors, windows), obstacles, required furniture set, and any fixed placements. The top-level `room-layout-*.json` files are examples you can copy here for batch runs. ([GitHub][1])
* **Tip:** Keep inputs normalized (units, naming, clockwise wall order), so all methods can parse them consistently.

### `constraints/`

* **Purpose:** Centralized **design rules**: clearances, wheelchair turning radii, min distances from doors/windows/radiators, reach ranges, power-socket adjacency, etc.
* **How it’s used:** Both Traditional and RL approaches should import these parameters to validate placements and score layouts.

### `Traditional/`

* **Purpose:** Baseline algorithms: constructive placement + heuristics + local improvement.
* **Typical pipeline:**

  1. **Seeding:** Place anchor items (bed, sofa) using wall adjacency + door/window offsets.
  2. **Feasibility checks:** Overlap, egress width, door swing, window access.
  3. **Local search:** Swap, nudge, rotate; accept changes that increase a multi-objective score.
* **When to use:** Fast, interpretable baselines; good for ablation studies on constraints.

### `Reinforcement-Learning/`

* **Purpose:** Learn a **policy** to place/move/rotate items sequentially to maximize a reward (layout score).
* **Typical components:**

  * **State:** Partial layout (grid or continuous), room boundary, openings, remaining furniture.
  * **Actions:** Place next item (x, y, θ), or adjust/move selected item.
  * **Reward:** Dense step rewards for feasibility + global terminal reward for aesthetics/accessibility.
  * **Training:** Curriculum from `room-layout-min.json` up to complex rooms.
* **When to use:** Discover non-obvious arrangements; adapt to new room archetypes.

### `Fine-Tuned/`

* **Purpose:** ML models **fine-tuned** to predict placements or to score layouts (learned evaluator).
* **Typical roles:**

  * **Predictor:** Given shell + openings + furniture set → suggest approximate anchors.
  * **Scorer:** Given a candidate layout → return a scalar “quality” used by search or RL.
* **When to use:** As a warm start for Traditional/RL or as a learned critic to improve selection.

### `Outputs/`

* **Purpose:** Persistent record of generated layouts and visualizations.
* **Includes:** JSON exports comparable to `optimized_barrier_free_layout.json`, and PNGs like `best_barrier_free_layout.png` for quick review. ([GitHub][1])

### `untrack/`

* **Purpose:** Workspace ignored by Git (e.g., heavy logs, cache, local runs).

---

## Data model: room & layout JSON

The repository includes multiple `room-layout-*.json` files and an `optimized_barrier_free_layout.json`. These strongly suggest a simple, portable JSON schema for both **input rooms** and **output layouts**. Below is a schema you can document in the repo to make expectations explicit (adjust field names if yours differ):

```json
{
  "meta": {
    "id": "room-001",
    "units": "m"
  },
  "room": {
    "polygon": [[0,0],[5,0],[5,4],[0,4]],
    "doors": [{"position":[0,2],"width":0.9,"swing":"in","wall":"W"}],
    "windows": [{"position":[3,4],"width":1.2,"sill":0.9}],
    "obstacles": [{"polygon":[[2.4,0],[2.6,0],[2.6,0.2],[2.4,0.2]], "type":"radiator"}]
  },
  "furniture": [
    {"name":"bed", "size":[2.0,1.4], "min_clearance":[0.75,0.5], "rotate": true},
    {"name":"wardrobe", "size":[1.6,0.6], "rotate": true},
    {"name":"desk", "size":[1.2,0.6], "power_required": true}
  ],
  "placement": [
    {"name":"bed", "xy":[0.5,0.5], "theta": 0.0},
    {"name":"wardrobe", "xy":[3.0,0.5], "theta": 1.57}
  ]
}
```

* **Inputs** must at least define `room` and the required `furniture` set.
* **Outputs** add a `placement` array containing final positions and rotations.
* **Barrier-free fields** (wheelchair turning circles, corridor width goals) can live in `constraints/` or under `meta.accessibility`.

---

## Scoring & constraints (recommended)

Use a **weighted multi-objective** score so approaches can be compared consistently:

1. **Feasibility (hard):** No overlaps; door swings clear; windows accessible; walkable graph connectivity.
2. **Accessibility:** Wheelchair turning radius; min corridor width; reach to switches/outlets; transfer space by bed/sofa.
3. **Functionality:** Task adjacency (desk↔outlet/window), TV↔sofa sightline, bed↔wardrobe proximity.
4. **Aesthetics (soft):** Alignment to walls; symmetry; balanced negative space.
5. **Penalties:** For violations; large penalties for hard constraints.

Document these weights in `constraints/weights.json` (or similar) so all methods share the same objective.

---

## Typical workflows

### 1) Baseline (Traditional)

```
python Traditional/run.py \
  --input Input-Layouts/room-layout-1.json \
  --constraints constraints/barrier_free.yaml \
  --out Outputs/layout_traditional_room1.json
```

* Starts with heuristic anchors, applies local search, writes JSON + PNG.

### 2) Reinforcement Learning

```
python Reinforcement-Learning/train.py \
  --rooms Input-Layouts \
  --constraints constraints/barrier_free.yaml \
  --runs 100000 \
  --save Reinforcement-Learning/checkpoints

python Reinforcement-Learning/eval.py \
  --input Input-Layouts/room-layout-3.json \
  --checkpoint Reinforcement-Learning/checkpoints/best.pt \
  --out Outputs/layout_rl_room3.json
```

* Trains an agent on the shared reward; evaluates on held-out rooms.

### 3) Fine-Tuned scoring + search

```
python Fine-Tuned/score.py --in Outputs/layout_traditional_room1.json
python Fine-Tuned/score_batch.py --in Outputs --pattern "*.json"
```

* Learned scorer ranks candidates and can be plugged into Traditional/RL loops.

> Exact script names may differ; adapt the commands to your file names.

---

## Reproducing the included example

The repo ships with:

* `optimized_barrier_free_layout.json`: a reference solution
* `best_barrier_free_layout.png`: its visualization for quick review ([GitHub][1])

**To reproduce:**

1. Run any approach on `room-layout-*.json`.
2. Compare your output score with the reference `optimized_barrier_free_layout.json`.
3. Use the visualization tool (e.g., `viz.py`) to render a PNG and compare with `best_barrier_free_layout.png`.

---

## Installation

* **Python:** 3.9+ recommended
* **Dependencies:**

  * `numpy`, `scipy`, `shapely` (geometry/overlap)
  * `networkx` (walkability graph)
  * `matplotlib` or `plotly` (visualization)
  * `pyyaml` or `pydantic` (config/schema)
  * For RL: `torch` or `tensorflow`, plus `gymnasium`/`stable-baselines3` (if used)

```
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
# or
pip install numpy shapely networkx matplotlib pyyaml
```

---

## Evaluation protocol

1. **Rooms:** Use `Input-Layouts/` plus `room-layout-min.json` and `test.json` for smoke tests. ([GitHub][1])
2. **Metrics:**

   * % hard-constraint violations (target: 0)
   * Accessibility score (0–1)
   * Functional adjacency score (0–1)
   * Aesthetic score (0–1)
   * Total objective (weighted sum)
3. **Reporting:** Save per-room CSV in `Outputs/metrics.csv` and a summary table.

---

## Visualization

Provide a minimal renderer (e.g., `viz.py`) to:

* Draw the shell, doors/windows, obstacles
* Render furniture rectangles (with rotation) and labels
* Indicate access corridors, swing arcs, and wheelchair turning circles
* Export to `PNG` (as in `best_barrier_free_layout.png`) for side-by-side comparisons ([GitHub][1])

---

## Extending the project

* **More constraints:** Add country-specific standards (DIN 18040, ADA) as switchable profiles.
* **Furniture catalogs:** Allow parametrized families (e.g., bed 90/140/160 cm).
* **3D export:** Write `glTF`/`IFC` for downstream BIM/VR workflows.
* **Human-in-the-loop:** Simple UI to lock positions and re-optimize the rest.
* **Generalization:** Train on synthetic room distributions; evaluate OOD layouts.

---

## FAQ

**Q: Why three approaches?**
A: They complement each other. Traditional methods are fast and interpretable; RL discovers novel arrangements; Fine-Tuned models provide learned priors/scores.

**Q: How is “barrier-free” enforced?**
A: As **hard constraints** (clearances, door swings) and **soft rewards** (maneuvering spaces) with high penalties for violations, ensuring wheelchair accessibility.

**Q: Can I plug in my own room?**
A: Yes—drop a JSON in `Input-Layouts/` matching the schema above and run any pipeline.

---

## License

MIT License. See `LICENSE` in the repo. ([GitHub][1])

---

## A note on provenance

This documentation reflects the **actual repository layout and filenames** visible on GitHub at the time of writing (folders: *Traditional*, *Reinforcement-Learning*, *Fine-Tuned*, *Input-Layouts*, *Outputs*, *constraints*, *untrack*; files including `best_barrier_free_layout.png`, `optimized_barrier_free_layout.json`, and multiple `room-layout-*.json`). ([GitHub][1])

---

[1]: https://github.com/nikhilgat/SmartFurnitureLayoutPlanner "GitHub - nikhilgat/SmartFurnitureLayoutPlanner"
