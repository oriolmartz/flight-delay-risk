# Candidate benchmark

This benchmark compares four model families on the same later validation block using ordered historical encoding. It is retained as model-development evidence; the public v1.0 artifact is chosen with temporal-stability evidence rather than this single split alone.

| Candidate | ROC-AUC | PR-AUC | Lift@10% | Raw Brier | Raw ECE |
|---|---:|---:|---:|---:|---:|
| `extra_trees` | 0.6430 | 0.2559 | 1.779× | 0.2463 | 0.3326 |
| `logistic_l1` | 0.6404 | 0.2516 | 1.765× | 0.2786 | 0.3644 |
| `random_forest` | 0.6439 | 0.2516 | 1.763× | 0.2502 | 0.3381 |
| `baseline` | 0.6338 | 0.2473 | 1.730× | 0.2795 | 0.3618 |

Single-split validation winner: **extra_trees**.
