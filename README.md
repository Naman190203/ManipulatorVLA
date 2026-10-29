# Manipulator VLA project

Run the commands below from the project root.

## Project layout

- `src/` contains the simulation bridge, bridge client, task generator, dataset
  recorder, and autonomous collector.
- `assets/` contains the MuJoCo scene, robot descriptions, dashboard, and other
  scene assets.
- `docs/` contains the bridge and dataset collection guides.
- `tools/` contains utilities for rebuilding or converting robot/scene assets.
- Generated demonstration runs are stored separately from the source code,
  including `vla_dataset_autonomous/`.

## Quick start

Install bridge dependencies and start the simulator:

```powershell
python -m pip install -r requirements-bridge.txt
python src/sim_bridge.py
```

The dashboard is served at `http://127.0.0.1:8765/`. For dataset collection,
see [docs/VLA_DATASET.md](docs/VLA_DATASET.md). For bridge details, see
[docs/BRIDGE.md](docs/BRIDGE.md).
