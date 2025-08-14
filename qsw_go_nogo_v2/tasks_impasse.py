import numpy as np
from math import log2

class ImpasseEscapeTask:
    """
    Two-phase task:
      Phase A: agent falls into a strong local attractor (decoy policy).
      Phase B: must escape the attractor to reach true goal.
    - We measure time-to-escape and success.
    """
    def __init__(self, seed=0, phase_len=12, context_window=3):
        self.rng = np.random.default_rng(seed)
        self.phase_len = phase_len
        self.context_window = context_window

        # Define decoy attractor sequence and true goal; share a prefix to induce stickiness
        prefix_len = 3
        self.prefix = self.rng.integers(0, 2, size=prefix_len).tolist()
        self.decoy = self.prefix + self.rng.integers(0, 2, size=phase_len-prefix_len).tolist()
        self.goal  = self.prefix + self.rng.integers(0, 2, size=phase_len-prefix_len).tolist()
        # Ensure goal != decoy
        while self.goal == self.decoy:
            self.goal  = self.prefix + self.rng.integers(0, 2, size=phase_len-prefix_len).tolist()

    def run(self, agent, gate_fn, max_steps=30):
        actions = []
        time_to_escape = None
        escaped = False
        # Phase A: agent tends to latch onto decoy due to Hebbian bumps
        # Escape is defined as producing a subsequence that diverges from decoy and matches goal's distinguishing bits.
        for t in range(max_steps):
            context = tuple(actions[-self.context_window:]) if actions else None
            gate_val = gate_fn(t, None)  # gating may depend on external uncertainty; keep simple here
            a, probs = agent.step(gate_val, context)
            actions.append(a)

            # Check for goal/decoy alignment over sliding window equal to phase_len
            if len(actions) >= self.phase_len:
                window = actions[-self.phase_len:]
                if not escaped:
                    if window[:3] == self.prefix:
                        # Check divergence from decoy after prefix AND convergence to goal bits
                        if window != self.decoy and window == self.goal:
                            escaped = True
                            time_to_escape = t + 1  # 1-indexed step count

                if window == self.goal:
                    return True, time_to_escape if time_to_escape is not None else t+1, tuple(actions)

        return False, time_to_escape if time_to_escape is not None else max_steps, tuple(actions)

def shannon_diversity(sequences):
    counts = {}
    for s in sequences:
        counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values()) or 1
    import numpy as np
    probs = [c/total for c in counts.values()]
    H = -sum(p * (0 if p==0 else np.log2(p)) for p in probs)
    return H, len(counts)
