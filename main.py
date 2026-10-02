"""
Online stable matching: babies and daycares. Final Assignment: Algorithmic Decision Making
Can & Femke

match[b] = daycare number of baby b, or -1 if baby b has no place.
Babies arrive in the order 0, 1, 2, ...
"""
import math
import random

# ---------------------------------------------------------------
# Step 1: the model
# ---------------------------------------------------------------
def make_instance(n, q, L):
    """
    n = number of daycares
    q = capacity of every daycare
    L = load (babies / total spots available)
    """
    Q = n * q                      # total spots available
    m = math.ceil(L * Q)           # number of babies

    # random spots in the 1x1 square, each spot is (x, y)
    daycare_pos = [(random.random(), random.random()) for _ in range(n)]
    baby_pos = [(random.random(), random.random()) for _ in range(m)]

    # dist[b][d] = distance from baby b to daycare d, divided by sqrt(2)
    # so that every distance is between 0 and 1
    dist = []

    for b in range(m):
        row = []
        for d in range(n):
            dx = baby_pos[b][0] - daycare_pos[d][0]
            dy = baby_pos[b][1] - daycare_pos[d][1]
            row.append(math.sqrt(dx**2 + dy**2) / math.sqrt(2))
        dist.append(row)

    baby_order = []   # baby_order[b] = daycares of baby b, favorite first
    baby_rank = []    # baby_rank[b][d] = 1 if d is the favorite of b, 2 if second, ...
    for b in range(m):
        order = sorted(range(n), key=lambda d: dist[b][d])
        rank = [0] * n
        for position, d in enumerate(order):
            rank[d] = position + 1
        baby_order.append(order)
        baby_rank.append(rank)

    return {"n": n, "q": q, "m": m, "dist": dist,
            "baby_order": baby_order, "baby_rank": baby_rank}


# ---------------------------------------------------------------
# Step 2: the measures
# ---------------------------------------------------------------
def evaluate(inst, match):
    n, q, m = inst["n"], inst["q"], inst["m"]
    dist, baby_rank = inst["dist"], inst["baby_rank"]

    # 1. matching size and 2. average rank
    size = 0
    rank_sum = 0
    for b in range(m):
        if match[b] != -1:
            size += 1
            rank_sum += baby_rank[b][match[b]]
    avg_rank = rank_sum / size

    # For each daycare: how many babies it has, and its farthest baby
    load = [0] * n
    worst = [-1] * n               # distance of its farthest baby
    for b in range(m):
        d = match[b]
        if d != -1:
            load[d] += 1
            worst[d] = max(worst[d], dist[b][d])

    # 3. blocking pairs: baby b and daycare d block if
    #    - b likes d more than her own match (or has no match), AND
    #    - d has a free place, or d likes b more than its farthest baby
    blocking = 0
    for b in range(m):
        if match[b] == -1:
            my_rank = n + 1        # unmatched: worse than any daycare
        else:
            my_rank = baby_rank[b][match[b]]
        for d in range(n):
            if baby_rank[b][d] < my_rank:
                if load[d] < q or dist[b][d] < worst[d]:
                    blocking += 1

    return size, avg_rank, blocking


# ---------------------------------------------------------------
# Step 3: baseline
# ---------------------------------------------------------------
def online_greedy(inst):
    """Every baby takes her favorite daycare that still has a free place."""
    n, q, m = inst["n"], inst["q"], inst["m"]
    load = [0] * n
    match = [-1] * m
    for b in range(m):                       # babies arrive one by one
        for d in inst["baby_order"][b]:      # favorite first
            if load[d] < q:
                match[b] = d
                load[d] += 1
                break
    return match


# ---------------------------------------------------------------
# Experiment: run many random worlds and take the average
# ---------------------------------------------------------------
def run_experiment(algorithms, n, q, L, runs=1000):
    totals = {name: [0, 0, 0] for name in algorithms}
    for _ in range(runs):
        inst = make_instance(n, q, L)
        for name, algo in algorithms.items():
            size, avg_rank, blocking = evaluate(inst, algo(inst))
            totals[name][0] += size
            totals[name][1] += avg_rank
            totals[name][2] += blocking
    for name in algorithms:
        size, avg_rank, blocking = [t / runs for t in totals[name]]
        print(f"  {name:22s} size={size:6.1f}  avg_rank={avg_rank:5.2f}  blocking={blocking:6.1f}")


if __name__ == "__main__":
    random.seed(123)              # same random numbers every run
    algorithms = {"Greedy (online)": online_greedy}

    for L in (0.8, 1.0, 1.25):
        print(f"\nn=10, q=8, L={L}")
        run_experiment(algorithms, n=10, q=8, L=L)