import sys, pandas as pd, numpy as np

def permutation_test(vals_a, vals_b, n_perm=5000, metric=lambda x: np.mean(x)):
    obs = metric(vals_a) - metric(vals_b)
    pooled = np.concatenate([vals_a, vals_b])
    rng = np.random.default_rng(0)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        a = pooled[:len(vals_a)]
        b = pooled[len(vals_a):]
        if (metric(a) - metric(b)) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)

def main(csv_path):
    df = pd.read_csv(csv_path)
    print("\n=== Summary (v2) ===")
    print(df)

    # Compare quantum vs pseudo for success_rate, avg_time_to_escape, diversity_bits, MI
    try:
        for metric_name in ["success_rate", "avg_time_to_escape", "diversity_bits", "early_gate_success_MI"]:
            q = df.loc[df["condition"]=="quantum", metric_name].values
            p = df.loc[df["condition"]=="pseudo", metric_name].values
            if len(q)>0 and len(p)>0:
                # For time_to_escape, lower is better; invert sign for comparison
                if metric_name == "avg_time_to_escape":
                    obs, pval = permutation_test(-q, -p)
                else:
                    obs, pval = permutation_test(q, p)
                print(f"Quantum-Pseudo {metric_name} diff={obs:.4f}, p~{pval:.4f}")
    except Exception as e:
        print("Permutation test error:", e)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "results/latest_results.csv")
