import csv
import sys
import time
from heapq import heappush, heappop


def parse_csv(filename):
    graph = {}
    stop_coords = {}
    with open(filename, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = row["start_stop"]
            end = row["end_stop"]
            line = row["line"]
            company = row["company"]
            start_lat = float(row["start_stop_lat"])
            start_lon = float(row["start_stop_lon"])
            end_lat = float(row["end_stop_lat"])
            end_lon = float(row["end_stop_lon"])
            graph.setdefault(start, []).append({
                'end_stop': end,
                'line': line,
                'company': company
            })
            stop_coords[start] = (start_lat, start_lon)
            stop_coords[end] = (end_lat, end_lon)

    return graph, stop_coords


def changes_heuristic(current_stop, end_stop):
    # we can calculate it manually also, for now it is 0 here
    return 0


def astar_changes(graph, coords, start_stop, end_stop):
    from collections import defaultdict
    INF = 10 ** 9
    g_scores = defaultdict(lambda: INF)
    f_scores = defaultdict(lambda: INF)
    start_state = (start_stop, None)
    g_scores[start_state] = 0
    f_scores[start_state] = 0 + changes_heuristic(start_stop, end_stop)
    parent = {}
    open_set = []
    heappush(open_set, (f_scores[start_state], start_state))
    while open_set:
        current_f, (current_stop, current_line) = heappop(open_set)

        if current_stop == end_stop:
            break

        if current_f > f_scores[(current_stop, current_line)]:
            continue

        if current_stop not in graph:
            continue

        for edge in graph[current_stop]:
            neighbor_stop = edge['end_stop']
            neighbor_line = edge['line']
            cost_of_change = 0 if (current_line == neighbor_line or current_line is None) else 1
            tentative_g = g_scores[(current_stop, current_line)] + cost_of_change
            neighbor_state = (neighbor_stop, neighbor_line)
            if tentative_g < g_scores[neighbor_state]:
                g_scores[neighbor_state] = tentative_g
                h_val = changes_heuristic(neighbor_stop, end_stop)
                f_scores[neighbor_state] = tentative_g + h_val
                parent[neighbor_state] = (current_stop, current_line)
                heappush(open_set, (f_scores[neighbor_state], neighbor_state))

    best_line = None
    best_cost = INF
    for (stp, ln) in g_scores:
        if stp == end_stop and g_scores[(stp, ln)] < best_cost:
            best_cost = g_scores[(stp, ln)]
            best_line = ln

    if best_cost == INF:
        return None, None

    path = []
    current_state = (end_stop, best_line)
    while current_state in parent:
        (cs, cl) = current_state
        prev_stop, prev_line = parent[current_state]
        path.append((cl, prev_stop, cs))
        current_state = (prev_stop, prev_line)

    path.reverse()
    return path, best_cost


def print_solution(path, total_changes):
    if not path:
        print("No route found.")
        return

    print("=== Travel Schedule (A* changes-based) ===")
    for (line, from_stop, to_stop) in path:
        if line is None:
            print(f"Start from {from_stop} => {to_stop}")

        else:
            print(f"Line {line}, from {from_stop} => {to_stop}")

    print(f"Total line changes: {total_changes}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python a_star_transfer_criteria.py <start> <end> <p>")
        sys.exit(1)

    start_stop = sys.argv[1]
    end_stop = sys.argv[2]
    criterion = sys.argv[3]
    if criterion != 'p':
        print("This script is for A* changes-based. Use 'p' as criterion.")
        sys.exit(1)

    comp_start = time.time()
    graph, coords = parse_csv("connection_graph.csv")
    path, cost = astar_changes(graph, coords, start_stop, end_stop)
    comp_end = time.time()
    print_solution(path, cost)
    sys.stderr.write(f"Total line changes = {cost}\n")
    sys.stderr.write(f"Computation time = {comp_end - comp_start:.4f} s\n")


if __name__ == "__main__":
    main()