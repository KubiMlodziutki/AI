import csv
import sys
import time
from heapq import heappush, heappop
import math


def parse_csv(filename):
    graph = {}
    stop_coords = {}
    with open(filename, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = row["start_stop"]
            end = row["end_stop"]
            dep_time = row["departure_time"]
            arr_time = row["arrival_time"]
            line = row["line"]
            company = row["company"]

            start_lat = float(row["start_stop_lat"])
            start_lon = float(row["start_stop_lon"])
            end_lat = float(row["end_stop_lat"])
            end_lon = float(row["end_stop_lon"])

            # We'll store edges by stops, ignoring actual times for the "changes-based" approach
            # but we do need them for identifying if we remain on the same line or if we switch lines.

            graph.setdefault(start, []).append({
                'end_stop': end,
                'line': line,
                'company': company
            })

            # coords for a potential heuristic, though for "changes-based"
            # we might just do a simple "straight-line" or zero heuristic
            stop_coords[start] = (start_lat, start_lon)
            stop_coords[end] = (end_lat, end_lon)

    return graph, stop_coords


def changes_heuristic(current_stop, end_stop, coords):
    # Possibly we can guess the number of line changes.
    # A trivial heuristic is 0 => we get D' BFS.
    # Or we can do distance-based guess.
    # We'll keep it 0 to ensure correctness (an admissible heuristic).
    return 0


def astar_changes(graph, coords, start_stop, end_stop):
    """
    We'll do an A* that tries to minimize the number of line changes.
    We'll store state as (stop, line) to keep track if we changed lines.
    So g((stop, line)) = how many times we have changed lines from the start.
    """
    from collections import defaultdict
    INF = 10 ** 9

    # We'll keep a dictionary: g_scores[(stop, line)] = minimal line changes so far
    g_scores = defaultdict(lambda: INF)
    f_scores = defaultdict(lambda: INF)

    # We can start with any line at the start_stop, but let's define a "no line" for start
    # so we don't count a line change at the beginning.
    start_state = (start_stop, None)
    g_scores[start_state] = 0
    f_scores[start_state] = 0 + changes_heuristic(start_stop, end_stop, coords)

    parent = {}

    open_set = []
    heappush(open_set, (f_scores[start_state], start_state))

    visited_stops = set()  # If we only track stops, we might lose line info, so let's not do that

    while open_set:
        current_f, (current_stop, current_line) = heappop(open_set)

        if current_stop == end_stop:
            # reconstruct
            break

        if current_f > f_scores[(current_stop, current_line)]:
            continue

        if current_stop not in graph:
            continue

        for edge in graph[current_stop]:
            neighbor_stop = edge['end_stop']
            neighbor_line = edge['line']

            # cost if we remain on the same line => no new change
            # cost if we switch line => +1
            cost_of_change = 0 if (current_line == neighbor_line or current_line is None) else 1
            tentative_g = g_scores[(current_stop, current_line)] + cost_of_change

            neighbor_state = (neighbor_stop, neighbor_line)

            if tentative_g < g_scores[neighbor_state]:
                g_scores[neighbor_state] = tentative_g
                h_val = changes_heuristic(neighbor_stop, end_stop, coords)
                f_scores[neighbor_state] = tentative_g + h_val
                parent[neighbor_state] = (current_stop, current_line)
                heappush(open_set, (f_scores[neighbor_state], neighbor_state))

    # find which line at end_stop yields the minimal changes
    best_line = None
    best_cost = INF
    for (stp, ln) in g_scores:
        if stp == end_stop and g_scores[(stp, ln)] < best_cost:
            best_cost = g_scores[(stp, ln)]
            best_line = ln

    if best_cost == INF:
        return None, None

    # reconstruct
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
        # If line is None, that means we started traveling from the start_stop
        # We'll just skip printing that as a line
        if line is None:
            print(f"Start from {from_stop} => {to_stop}")
        else:
            print(f"Line {line}, from {from_stop} => {to_stop}")
    print(f"Total line changes: {total_changes}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python task3_astar_changes.py <start_stop> <end_stop> <criterion='p'>")
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
