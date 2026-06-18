# Project Aethelred

Adaptive Autonomous Swarm Intelligence System — **a research simulation and concept study**.

Aethelred explores a mother-drone tactical AI that commands an expendable swarm and
adapts to threats through a continual-learning loop. This repository is **simulation
only** — a 2D [Gymnasium](https://gymnasium.farama.org/) environment plus a PyTorch
policy stack. There is no hardware, actuation, or networking, and (per the concept
document) any real system of this kind would require robust human-in-the-loop controls
and clear rules of engagement.

## Contents

- [`aethelred/`](aethelred/) — the simulation, policy stack, training, and tests.
  See [`aethelred/README.md`](aethelred/README.md) for architecture and usage.
- `aethelred-concept.html` / `aethelred-concept.docx` — the concept document (v1.0).

## Quickstart

```bash
cd aethelred
pip install -e ".[dev]"

python scripts/run_simulation.py --steps 200 --seed 42   # watch a sim (random policy)
python scripts/train.py --mode online --episodes 100     # train the tactical AI (PPO)
pytest -q                                                 # run the test suite
```

## Status

The codebase has been audited and the core ML pipeline made functional: PPO trains the
policy end-to-end (train/inference parity), continual-learning adaptation works, the
swarm decomposes a high-level command into per-role behaviour with comms-loss autonomy,
and the deployment export path is tested. See [`aethelred/README.md`](aethelred/README.md)
for details, current results, and known limitations.
