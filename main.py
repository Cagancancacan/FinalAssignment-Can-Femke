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
def make_instance(n, q_min, q_max, L, random_prefs=True):
    """
    n = number of daycares
    q = capacity of every daycare
    L = load (babies / total spots available)
    """
    caps = [random.randint(q_min, q_max) for _ in range(n)]
    Q = sum(caps)
    m = math.ceil(L * Q)

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
            row.append(math.sqrt(dx ** 2 + dy ** 2) / math.sqrt(2))
        dist.append(row)

    baby_order = []  # baby_order[b] = daycares of baby b, favorite first
    baby_rank = []  # baby_rank[b][d] = 1 if d is the favorite of b, 2 if second, ...

    for b in range(m):
        if random_prefs:
            order = list(range(n))  # [0, 1, 2, ..., n-1]
            random.shuffle(order)  # put them in a random order
        else:
            order = sorted(range(n), key=lambda d: dist[b][
                d])  # we could use this in case the babies preferences are based on distance
        rank = [0] * n

        for position, d in enumerate(order):
            rank[d] = position + 1
        baby_order.append(order)  # Lists baby b's daycares from favorite to least favorite
        baby_rank.append(rank)  # Gives position of daycare d in baby b's ranking (1 is best)

    return {"n": n, "caps": caps, "m": m, "dist": dist,
            "baby_order": baby_order, "baby_rank": baby_rank}


# ---------------------------------------------------------------
# Step 2: the measures
# ---------------------------------------------------------------
def evaluate(inst, match):
    n, caps, m = inst["n"], inst["caps"], inst["m"]
    dist, baby_rank = inst["dist"], inst["baby_rank"]

    # 1. matching size and 2. average rank, 3. average distance (how close assigned babies live to their daycares)
    size = 0
    rank_sum = 0
    distance_sum = 0

    for b in range(m):
        if match[b] != -1:
            size += 1
            rank_sum += baby_rank[b][match[b]]
            distance_sum += dist[b][match[b]]

    avg_rank = rank_sum / size if size > 0 else float("nan")
    avg_distance = distance_sum / size if size > 0 else float("nan")

    # Fraction of all daycare places that are filled
    utilisation = size / sum(caps)

    # For each daycare: how many babies it has, and its farthest baby
    load = [0] * n
    worst = [-1] * n  # distance of its farthest baby

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
            my_rank = n + 1  # unmatched: worse than any daycare
        else:
            my_rank = baby_rank[b][match[b]]
        for d in range(n):
            if baby_rank[b][d] < my_rank:
                if load[d] < caps[d] or dist[b][d] < worst[d]:
                    blocking += 1

    return size, avg_rank, blocking, avg_distance, utilisation


# ---------------------------------------------------------------
# Step 3: baseline
# ---------------------------------------------------------------
def online_greedy(inst):
    """Every baby takes her favorite daycare that still has a free place."""
    n = inst["n"]
    caps = inst["caps"]
    m = inst["m"]

    load = [0] * n  # load[d] records how many babies are currently assigned to daycare d
    match = [-1] * m  # initially all babies are unmachted and thus equal to -1

    for b in range(m):  # babies arrive one by one
        for d in inst["baby_order"][b]:  # favorite first
            if load[d] < caps[d]:
                match[b] = d
                load[d] += 1
                break
    return match


# ---------------------------------------------------------------
# Step 4: baseline + distance threshold
# ---------------------------------------------------------------
def online_threshold(inst, thresh):
    """Same as greedy but a baby can only be matched to its preferred daycare if it is within the right distance. """
    n = inst["n"]
    caps = inst["caps"]
    m = inst["m"]
    dist = inst["dist"]

    load = [0] * n
    match = [-1] * m

    for b in range(m):  # babies arrive one by one
        for d in inst["baby_order"][b]:  # favorite first
            if load[d] < caps[d] and dist[b][d] < thresh:  # add dist threshold condition
                match[b] = d
                load[d] += 1
                break
    return match


# ---------------------------------------------------------------
# Experiment: run many random worlds and take the average
# ---------------------------------------------------------------
def run_experiment(algorithms, n, q_min, q_max, L, runs=1000, random_prefs=True):
    totals = {name: [0, 0, 0, 0, 0] for name in algorithms}

    for _ in range(runs):
        inst = make_instance(n, q_min, q_max, L, random_prefs=random_prefs)

        for name, algo in algorithms.items():
            size, avg_rank, blocking, avg_distance, utilisation = evaluate(inst, algo(inst))

            totals[name][0] += size / inst["m"]
            totals[name][1] += avg_rank
            totals[name][2] += blocking
            totals[name][3] += avg_distance
            totals[name][4] += utilisation

    for name in algorithms:
        rate, avg_rank, blocking, avg_distance, utilisation = [
            t / runs for t in totals[name]
        ]

        print(
            f"  {name:28s} "
            f"matched={rate:6.1%}  "
            f"avg_rank={avg_rank:5.2f}  "
            f"blocking_pairs={blocking:9.1f}  "
            f"avg_distance={avg_distance:5.3f}  "
            f"filled={utilisation:6.1%}"
        )

if __name__ == "__main__":
    random.seed(123)  # same random numbers every run

    threshold = 0.2
    algorithms = {"Greedy (online)": online_greedy,
                  }

    # Compare several distance thresholds
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.6, 1.0):
        algorithms[f"Threshold {threshold:.1f}"] = (
            lambda inst, t=threshold: online_threshold(inst, t)
        )

    for L in (0.8, 1.0, 1.25):
        print(f"\nn=100, q_min=8, q_max=25, L={L}")
        run_experiment(algorithms, n=100, q_min=8, q_max=25, L=L, random_prefs=True)