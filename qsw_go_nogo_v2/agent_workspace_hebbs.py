import numpy as np

class WorkspaceAgentHebb:
    """
    Global-workspace-like agent with Hebbian bumping.
    - selection_temp (WTA gain) controls sharpness of choice.
    - On each context key, a winning action gets a small persistent bump.
    """
    def __init__(self, n_hidden=24, temperature=1.3, selection_temp=0.6, seed=0, hebb_gain=0.05):
        self.rng = np.random.default_rng(seed)
        self.h = self.rng.normal(0, 0.1, size=(n_hidden,))
        self.temperature = temperature
        self.selection_temp = selection_temp
        self.pref = self.rng.normal(0, 0.3, size=(2,))
        self.hebb = {}  # dict: context_key -> [bump0, bump1]
        self.hebb_gain = hebb_gain

    def _context_key(self, task_state):
        # task_state expected as a tuple of last 3 actions or similar
        return tuple(task_state) if task_state is not None else ('root',)

    def step(self, gate_val, task_state):
        # Exploit vs explore mix; gate controls explore weighting
        exploit_logits = self.pref + 0.2 * self.h[:2]
        explore_logits = self.rng.normal(0, 0.5, size=(2,))

        logits = (1 - gate_val) * exploit_logits + gate_val * explore_logits

        # Hebbian bump by context
        key = self._context_key(task_state)
        bump = self.hebb.get(key, np.zeros(2))
        logits = logits + bump

        # WTA selection temperature
        probs = np.exp(logits / self.selection_temp)
        probs = probs / probs.sum()

        action = 0 if self.rng.random() < probs[0] else 1

        # Update hidden state
        self.h = 0.95 * self.h + 0.05 * self.rng.normal(0, 0.1, size=self.h.shape)

        # Apply Hebbian bump to the chosen action for this context
        new_bump = bump.copy()
        new_bump[action] += self.hebb_gain
        # small decay to avoid runaway
        new_bump *= 0.995
        self.hebb[key] = new_bump

        return action, probs
