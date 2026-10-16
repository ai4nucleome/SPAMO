# Changelog

## 1.0.0
Repository release of the model used in the paper (previously the development package `spamoRev`).

* Standard `src/` layout, `pyproject.toml`, console script `spamo`, unit tests (`python -m unittest discover -s tests`).
* Feature-graph neighbourhood default **k = 20** (was 6 in the development presets); the cache loader checks that a cache
  was built with the requested k.
* New `spamo run` (preprocess + train + cluster) and `--clusters` override; `--seed` default is 2024.
* Evaluation reports the six paper metrics (NMI, AMI, FMI, ARI, F1, Jaccard) plus `Avg`/`Avg6`
  (`spamo.metrics`); the MB-specific "named subset" logic is replaced by the generic `--exclude-labels`.
* Class renamed `SPAMO` (`SpaMO` kept as alias); `scripts/benchmark.py` and `scripts/compare_runs.py` added.
* No change to the model, graph construction, loss, optimiser or training loop: the new package is bit-identical to the
  development package on CPU (see `docs/VALIDATION.md`).
