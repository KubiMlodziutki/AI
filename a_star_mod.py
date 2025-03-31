import csv
import sys
import math
import time as pytime
from collections import defaultdict
from math import inf
import heapq


def time_to_minutes(hhmmss: str) -> int:
    hh, mm, ss = hhmmss.split(':')
    return int(hh) * 60 + int(mm) + int(ss) // 60


def parse_csv(filename):
    graph = {}
    coords = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = row["start_stop"]
            end = row["end_stop"]
            dep = time_to_minutes(row["departure_time"])
            arr = time_to_minutes(row["arrival_time"])
            line = row["line"]

            slat, slon = float(row["start_stop_lat"]), float(row["start_stop_lon"])
            elat, elon = float(row["end_stop_lat"]), float(row["end_stop_lon"])
            coords[start] = (slat, slon)
            coords[end] = (elat, elon)

            graph.setdefault(start, []).append({
                'end_stop': end,
                'dep': dep,
                'arr': arr,
                'line': line
            })

    return graph, coords


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def combined_heuristic(current_stop, current_line, end_stop, coords, alpha, beta):
    if current_stop not in coords or end_stop not in coords:
        return 0

    lat1, lon1 = coords[current_stop]
    lat2, lon2 = coords[end_stop]
    dist_km = haversine_distance(lat1, lon1, lat2, lon2)
    time_est = dist_km * 2
    return alpha * time_est


def modified_astar(graph, coords, start_stop, end_stop, start_time, alpha, beta):
    g_score = defaultdict(lambda: inf)
    time_score = defaultdict(lambda: inf)
    changes_score = defaultdict(lambda: inf)
    parent = {}
    start_state = (start_stop, None)
    g_score[start_state] = 0
    time_score[start_state] = start_time
    changes_score[start_state] = 0

    def f_score(state):
        (stp, ln) = state
        return g_score[state] + combined_heuristic(stp, ln, end_stop, coords, alpha, beta)

    open_heap = []
    heapq.heappush(open_heap, (f_score(start_state), start_state))

    while open_heap:
        _, (cur_stop, cur_line) = heapq.heappop(open_heap)
        if cur_stop == end_stop:
            break

        cur_time = time_score[(cur_stop, cur_line)]
        cur_changes = changes_score[(cur_stop, cur_line)]
        if cur_stop not in graph:
            continue

        for edge in graph[cur_stop]:
            neighbor_stop = edge['end_stop']
            neighbor_line = edge['line']
            dep = edge['dep']
            arr = edge['arr']

            if dep >= cur_time:
                new_time = arr
                line_change = 0 if (cur_line == neighbor_line or cur_line is None) else 1
                new_changes = cur_changes + line_change
                new_cost = alpha * (new_time - start_time) + beta * new_changes
                neighbor_state = (neighbor_stop, neighbor_line)
                if new_cost < g_score[neighbor_state]:
                    g_score[neighbor_state] = new_cost
                    time_score[neighbor_state] = new_time
                    changes_score[neighbor_state] = new_changes
                    parent[neighbor_state] = (cur_stop, cur_line)
                    heapq.heappush(open_heap, (
                        new_cost + combined_heuristic(neighbor_stop, neighbor_line, end_stop, coords, alpha, beta),
                        neighbor_state))

    best_cost = inf
    best_time = None
    best_changes = None
    best_line = None

    for (stp, ln) in g_score:
        if stp == end_stop and g_score[(stp, ln)] < best_cost:
            best_cost = g_score[(stp, ln)]
            best_line = ln
            best_time = time_score[(stp, ln)]
            best_changes = changes_score[(stp, ln)]

    if best_cost == inf:
        return [], inf, None, None

    path = []
    cur = (end_stop, best_line)
    while cur in parent:
        cstop, cline = cur
        pstop, pline = parent[cur]
        path.append((cline, pstop, cstop))
        cur = (pstop, pline)

    path.reverse()
    return path, best_cost, best_time, best_changes


def minutes_to_time_str(m: int) -> str:
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}:00"


def print_solution(path, total_cost, arrival_time, changes):
    if not path:
        print("No route found")
        return

    print("=== Modified A* solution ===")
    for (line, from_stop, to_stop) in path:
        print(f"Line {line}, from {from_stop} => {to_stop}")

    print(f"Final combined cost: {total_cost:.2f}")
    if arrival_time is not None:
        print(f"Arrival time: {minutes_to_time_str(arrival_time)}")

    if changes is not None:
        print(f"Line changes: {changes}")


def main():
    if len(sys.argv) < 6:
        print("Usage: python a_star_mod.py <start> <end> <start_time> <alpha> <beta>")
        sys.exit(1)

    start_stop = sys.argv[1]
    end_stop = sys.argv[2]
    start_time = time_to_minutes(sys.argv[3])
    alpha = float(sys.argv[4])
    beta = float(sys.argv[5])
    comp_start = pytime.time()
    graph, coords = parse_csv("connection_graph.csv")
    path, cost, arr_time, chg = modified_astar(graph, coords, start_stop, end_stop, start_time, alpha, beta)
    comp_end = pytime.time()
    print_solution(path, cost, arr_time, chg)
    sys.stderr.write(f"Computation time = {comp_end - comp_start:.4f} s\n")


if __name__ == "__main__":
    main()