import sys
import csv
import random
from math import inf
from collections import defaultdict
from heapq import heappush, heappop


def time_to_minutes(hhmmss: str) -> int:
    hh, mm, ss = hhmmss.split(':')
    return int(hh) * 60 + int(mm) + int(ss) // 60


def minutes_to_time_str(m: int) -> str:
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}:00"


def read_stdin():
    A = sys.stdin.readline().strip()
    L_line = sys.stdin.readline().strip()
    crit = sys.stdin.readline().strip()
    start_time_str = sys.stdin.readline().strip()
    L = [x.strip() for x in L_line.split(';') if x.strip()]
    return A, L, crit, start_time_str


def parse_csv(filename="connection_graph.csv"):
    graph = defaultdict(list)
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = row["start_stop"]
            end = row["end_stop"]
            dep = time_to_minutes(row["departure_time"])
            arr = time_to_minutes(row["arrival_time"])
            ln = row["line"]
            graph[start].append({
                'end_stop': end,
                'dep': dep,
                'arr': arr,
                'line': ln
            })

    return dict(graph)


def dijkstra_time_fullpath(graph, start_stop, end_stop, start_time):
    INF = 10 ** 15
    dist = {start_stop: start_time}
    parent = {}
    pq = []
    heappush(pq, (start_time, start_stop))

    while pq:
        curr_t, stp = heappop(pq)
        if stp == end_stop:
            break

        if curr_t > dist.get(stp, INF):
            continue

        if stp not in graph:
            continue

        for edge in graph[stp]:
            nxt = edge['end_stop']
            dep = edge['dep']
            arr = edge['arr']
            ln = edge['line']
            if dep >= curr_t:
                if arr < dist.get(nxt, INF):
                    dist[nxt] = arr
                    parent[nxt] = (stp, ln, dep, arr)
                    heappush(pq, (arr, nxt))

    if end_stop not in dist:
        return [], None

    edges_rev = []
    cur = end_stop
    while cur != start_stop:
        if cur not in parent:
            return [], None

        pstop, used_line, depT, arrT = parent[cur]
        edges_rev.append((used_line, depT, pstop, arrT, cur))
        cur = pstop

    edges_rev.reverse()
    return edges_rev, dist[end_stop]


def dijkstra_changes_fullpath(graph, start_stop, end_stop):
    cost = defaultdict(lambda: inf)
    parent = {}
    cost[(start_stop, None)] = 0
    heappush(pq := [], (0, start_stop, None))

    while pq:
        zm, stp, line = heappop(pq)
        if stp == end_stop:
            break

        if zm > cost[(stp, line)]:
            continue

        if stp not in graph:
            continue

        for edge in graph[stp]:
            nxt = edge['end_stop']
            ln = edge['line']
            add_chg = 0 if (line is None or ln == line) else 1
            newch = zm + add_chg
            if newch < cost[(nxt, ln)]:
                cost[(nxt, ln)] = newch
                parent[(nxt, ln)] = (stp, line)
                heappush(pq, (newch, nxt, ln))

    best_line = None
    best_val = inf
    for (s, l) in cost:
        if s == end_stop and cost[(s, l)] < best_val:
            best_val = cost[(s, l)]
            best_line = l

    if best_val == inf:
        return [], None

    route_rev = []
    cur = (end_stop, best_line)
    while cur in parent:
        route_rev.append(cur)
        pstop, pln = parent[cur]
        cur = (pstop, pln)

    route_rev.append(cur)
    route_rev.reverse()
    edges = []
    current_time = 0
    for i in range(len(route_rev) - 1):
        (stopA, lineA) = route_rev[i]
        (stopB, lineB) = route_rev[i + 1]
        lnB = lineB if lineB else "?"
        depT = current_time
        arrT = current_time + 1
        edges.append((lnB, depT, stopA, arrT, stopB))
        current_time += 1

    return edges, best_val


def build_tsp_cost_matrix(all_stops, graph, criterion):
    n = len(all_stops)
    INF = 10 ** 9
    cost_mat = [[INF] * n for _ in range(n)]
    for i in range(n):
        cost_mat[i][i] = INF

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            A = all_stops[i]
            B = all_stops[j]
            if criterion == 't':
                edges, final_arr = dijkstra_time_fullpath(graph, A, B, 0)
                if edges and final_arr is not None:
                    cost_mat[i][j] = final_arr

            else:
                edges, val = dijkstra_changes_fullpath(graph, A, B)
                if edges and val is not None:
                    cost_mat[i][j] = val

    return cost_mat


def route_cost_TSP(route, cost_mat):
    if not route:
        return 10 ** 9

    s = cost_mat[0][route[0]]
    for i in range(len(route) - 1):
        s += cost_mat[route[i]][route[i + 1]]

    s += cost_mat[route[-1]][0]
    return s


def generate_neighbors(route):
    neighbors = []
    n = len(route)
    for i in range(n - 1):
        for j in range(i + 1, n):
            nr = route[:]
            nr[i], nr[j] = nr[j], nr[i]
            neighbors.append(nr)
    return neighbors


def tabu_search_unlimited(cost_mat, subN, max_iter=500):
    route = list(range(1, subN + 1))
    random.shuffle(route)
    best_route = route[:]
    best_cost = route_cost_TSP(route, cost_mat)
    T = set()
    T.add(tuple(route))

    for _ in range(max_iter):
        neighs = generate_neighbors(route)
        best_nb = None
        best_nb_cost = 10 ** 15
        for nb in neighs:
            nb_t = tuple(nb)
            if nb_t in T:
                continue

            c = route_cost_TSP(nb, cost_mat)
            if c < best_nb_cost:
                best_nb_cost = c
                best_nb = nb

        if best_nb is None:
            break

        route = best_nb
        T.add(tuple(route))
        if best_nb_cost < best_cost:
            best_cost = best_nb_cost
            best_route = route[:]

    return best_route, best_cost


def build_full_schedule(route, all_stops, graph, criterion, user_start_time):
    final_edges = []
    current_time = time_to_minutes(user_start_time) if criterion == 't' else 0
    seq = [0] + route + [0]
    for i in range(len(seq) - 1):
        iA = seq[i]
        iB = seq[i + 1]
        A_stop = all_stops[iA]
        B_stop = all_stops[iB]
        if criterion == 't':
            edges_list, farr = dijkstra_time_fullpath(graph, A_stop, B_stop, current_time)
            if not edges_list:
                continue

            final_edges.extend(edges_list)
            if farr is not None:
                current_time = farr

        else:
            edges_list, val = dijkstra_changes_fullpath(graph, A_stop, B_stop)
            if not edges_list:
                continue

            shifted = []
            for (ln, depT, fr, arrT, to) in edges_list:
                shifted.append((ln, depT + current_time, fr, arrT + current_time, to))

            final_edges.extend(shifted)
            current_time += len(edges_list)

    return final_edges


def print_final_schedule(final_edges):
    if not final_edges:
        print("No route found.")
        return

    print("=== Final Tabu TSP Schedule ===")
    current_line = None
    block_start_dep = None
    block_from_stop = None
    block_end_arr = None
    block_to_stop = None

    def flush_block():
        if current_line is not None:
            dep_str = minutes_to_time_str(block_start_dep)
            arr_str = minutes_to_time_str(block_end_arr)
            print(f"Line {current_line}, board at {dep_str} from {block_from_stop}, arrive at {arr_str} in "
                  f"{block_to_stop}")

    for i, (ln, dep, fr, arr, to) in enumerate(final_edges):
        if ln == current_line:
            block_end_arr = arr
            block_to_stop = to

        else:
            flush_block()
            current_line = ln
            block_start_dep = dep
            block_from_stop = fr
            block_end_arr = arr
            block_to_stop = to

    flush_block()


def main():
    A, L, crit, start_time_str = read_stdin()
    graph = parse_csv("connection_graph.csv")
    all_stops = [A] + L
    cost_mat = build_tsp_cost_matrix(all_stops, graph, crit)
    subN = len(all_stops) - 1
    best_route, best_cost = tabu_search_unlimited(cost_mat, subN, max_iter=500)
    edges = build_full_schedule(best_route, all_stops, graph, crit, start_time_str)
    print_final_schedule(edges)


if __name__ == "__main__":
    main()