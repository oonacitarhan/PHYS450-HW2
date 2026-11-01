# Quantum Information Calculations

Python / QuTiP exercises on **quantum coherence, correlations, and entanglement**, originally written for PHYS 450 Homework 2.

## Topics

[solutions.py](solutions.py) compares calculations for a pure two-qubit state and a related mixed state:

| Exercises | Calculation |
| --- | --- |
| `q1`, `q5` | Relative entropy of coherence and the l1 norm of coherence |
| `q2`, `q6` | Total correlations for the state and its diagonal counterpart |
| `q3`, `q7` | Entanglement of formation |
| `q4`, `q8` | Entanglement negativity using manually constructed partial transposes |

## Repository contents

- **[solutions.py](solutions.py)** — state construction and calculations for questions 1–8.
- **[q9.png](q9.png)** and **[q10.png](q10.png)** — accompanying image files.

## Running the script

The script imports **QuTiP, NumPy, and SciPy**. With a compatible environment installed, run:

```bash
python solutions.py
```

This is historical coursework from 2021. Dependencies were not pinned, and the script uses older QuTiP / SciPy interfaces such as `.data.toarray()` and `scipy.sparse.dia`. Running it with current releases may require compatibility changes.

## Scope

These are worked examples for specific states, rather than a reusable quantum-information library. Partial transposes are entered manually, and numerical outputs have not been revalidated as part of the documentation refresh.
