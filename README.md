# Quantum Go/No-Go Workspace (v2.5)

**A research-oriented platform for testing quantum randomness in adaptive agent decision-making.**  
This experiment uses the ANU Quantum Random Number Generator (QRNG) API alongside pseudo-random and deterministic baselines, comparing their effects on agent performance in an impasse escape task.

---

## Overview

This project implements a **go/no-go experimental paradigm** where an agent attempts to escape decision impasses under three distinct conditions:

- **Quantum:** Decisions gated by live ANU QRNG bits.  
- **Pseudo:** High-quality pseudo-random generator (NumPy MT19937).  
- **Deterministic:** Fixed, non-random sequence for gating.

The agent's ability to escape impasses is measured across hundreds of trials per condition, producing performance, diversity, and correlation metrics.

---

## Key Metrics

- **Success Rate** — Fraction of trials where the agent escaped the impasse within the time limit.  
- **Average Time to Escape** — Lower values indicate faster problem solving.  
- **Diversity (Shannon Bits)** — Measures variety in action sequences.  
- **Mutual Information (Early Gate ↔ Success)** — Captures predictive relationship between early gating and success.  
- **Quantum Ratio** — Proportion of true quantum bits successfully used (lower ratios indicate rate-limiting or fallback usage).

---

## Outputs

For each run, the app saves:

1. **Plots**  
   - Success Rate  
   - Average Time to Escape  
   - Diversity Bits  
   - Early Gate ↔ Success Mutual Information  

2. **Results Table** (CSV)  
3. **PDF Summary Report**  
4. **ZIP Archive of All Artifacts** (plots, CSV, report)

---

## Running the Experiment

1. **Launch the Streamlit App**: [Live Demo Here](https://gswv25-3annkcrumedcogwfu6xyk3.streamlit.app/)  
2. Select trial count, phase length, and conditions.  
3. *(Optional)* Run a **health check** to verify quantum source availability.  
4. Click **Run Experiment** to generate outputs.  

---

## Why It Matters

This experiment provides a reproducible framework for **evaluating the potential role of quantum randomness in decision-making systems**.  
By comparing against pseudo and deterministic baselines, collaborators can explore:

- Whether quantum randomness confers measurable advantages.  
- How agents adapt under different uncertainty regimes.  
- Opportunities for hybrid randomness strategies in AI control systems.

---

## License

MIT License — feel free to fork, adapt, and extend for your own research.

---
