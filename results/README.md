# Published results

Everything in this directory was produced from the **full dataset** (4,704
students), not from the synthetic sample in `data/`. These are the numbers and
figures that appear in the paper.

| File | Contents |
|---|---|
| `results_full.json` | Every reported statistic, machine-readable |
| `results_full.txt` | The same, as printed by `run_all.py` |
| `figures/Fig1.png` | Analytical pipeline |
| `figures/Fig2.png` | Effect of target leakage across the three configurations |
| `figures/Fig3.png` | ROC curves, corrected models |
| `figures/Fig4.png` | Confusion matrices, corrected models |
| `figures/Fig5.png` | Partial decision tree, thresholds in raw units |
| `figures/Fig6.png` | Feature importance, corrected models |

To regenerate, run `run_all.py` and `python -m src.figures` against the full
extract. Running them against the synthetic sample will produce different
numbers; see `data/README.md`.
