import argparse, csv, os
import numpy as np
from datetime import datetime
from qrng_client_v2 import QRNGClient
from agent_workspace_hebbs import WorkspaceAgentHebb
from tasks_impasse import ImpasseEscapeTask, shannon_diversity

def gate_policy(mode_source):
    src = QRNGClient(mode_source)
    def fn(t, probs=None):
        # Uncertainty-triggered gating if available
        if probs is not None:
            p0 = float(probs[0])
            if abs(p0 - 0.5) > 0.05:
                return 0.0
        val = src.next()
        return 1.0 if val*1.5 > 1.0 else val*1.5
    return fn

def run_trials(n_trials, condition, seed, phase_len=12):
    rng = np.random.default_rng(seed)
    successes = 0
    times = []
    sequences = []
    early_gates = []
    late_outcomes = []  # success indicator late in trajectory

    for i in range(n_trials):
        agent = WorkspaceAgentHebb(seed=int(rng.integers(0, 1e9)))
        task = ImpasseEscapeTask(seed=int(rng.integers(0, 1e9)), phase_len=phase_len)
        mode = "quantum" if condition == "quantum" else ("pseudo" if condition == "pseudo" else "deterministic")
        gp = gate_policy(mode)

        # Run with logging of early gates (first 4 steps)
        actions = []
        time_to_escape = None
        success = False
        max_steps = 30
        for t in range(max_steps):
            context = tuple(actions[-3:]) if actions else None
            # we simulate uncertainty: before stepping, we estimate probs by a dry run with gate=0 (approximate)
            # For simplicity, we won't dry run; we just call with None which uses only step-based gating.
            gate_val = gp(t, None)
            if t < 4: early_gates.append(gate_val)

            a, probs = agent.step(gate_val, context)
            actions.append(a)

            if len(actions) >= task.phase_len:
                window = actions[-task.phase_len:]
                if time_to_escape is None and window[:3] == task.prefix and window != task.decoy and window == task.goal:
                    time_to_escape = t + 1
                if window == task.goal:
                    success = True
                    break

        if success: successes += 1
        times.append(time_to_escape if time_to_escape is not None else max_steps)
        sequences.append(tuple(actions))
        late_outcomes.append(1 if success else 0)

    H, unique = shannon_diversity(sequences)
    # Mutual information between early gating (binned) and late success
    import numpy as np
    eg = np.array(early_gates)
    if len(eg) == 0:
        mi = 0.0
    else:
        # bin early gates into 5 bins
        bins = np.linspace(0, 1, 6)
        digitized = np.digitize(eg, bins) - 1
        # repeat late outcomes to match length (each trial contributed up to 4 gates)
        repeats = int(len(eg) / max(len(late_outcomes),1)) if len(late_outcomes)>0 else 0
        lo = np.repeat(late_outcomes, 4)[:len(eg)]
        from sklearn.metrics import mutual_info_score
        mi = float(mutual_info_score(digitized, lo))

    return {
        "condition": condition,
        "n_trials": n_trials,
        "success_rate": successes / max(n_trials,1),
        "avg_time_to_escape": float(np.mean(times)),
        "diversity_bits": H,
        "unique_seq": unique,
        "early_gate_success_MI": mi
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=1200)
    ap.add_argument("--conditions", nargs="+", default=["quantum","pseudo","deterministic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase_len", type=int, default=12)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join("results", f"results_{stamp}.csv")
    latest = os.path.join("results", "latest_results.csv")

    rows = []
    for cond in args.conditions:
        rows.append(run_trials(args.n_trials, cond, args.seed, args.phase_len))

    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(latest, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("Wrote:", out_csv)
    print("Also wrote:", latest)

if __name__ == "__main__":
    main()
