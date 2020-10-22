import numpy as np

# ps: np.array([n_points], dtype=np.float) (the probability f(x+delta) assigned to the recourse delta of x)
# e: float
# d: float
def compute_t(ps, e, d):
    # function that computes upper bound on threshold
    
    k = _compute_k(len(ps), e, d)
    return sorted(ps)[k]

def _compute_k(n, e, d):
    r = 0.0
    s = 0.0
    for h in range(n + 1):
        if h == 0:
            r = n * np.log(1.0 - e)
        else:
            r += np.log(n - h + 1) - np.log(h) + np.log(e) - np.log(1.0 - e)
        s += np.exp(r)
        if s > d:
            if h == 0:
                raise Exception('No valid threshold')
            else:
                return h - 1
    return n