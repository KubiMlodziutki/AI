import csv
import sys
import math
import time as pytime
from heapq import heappush, heappop

def time_to_minutes(hhmmss: str) -> int:
    hh, mm, ss = hhmmss.split(':')
    return int(hh)*60 + int(mm) + int(ss)//60

def minutes_to_time_str(m: int) -> str:
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}:00"

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

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

def astar_time(graph, coords, start_stop, end_stop, start_time):
    """
    A* minimalizująca czas.
    g_score[stop] = najwcześniejszy czas dotarcia
    f_score[stop] = g_score[stop] + heurystyka (szacowany koszt do celu)
    """
    def heuristic(st):
        if st not in coords or end_stop not in coords:
            return 0
        (lat1, lon1) = coords[st]
        (lat2, lon2) = coords[end_stop]
        # Zakładamy np. 2 min na 1 km (30 km/h)
        dist_km = haversine_distance(lat1, lon1, lat2, lon2)
        return dist_km * 2

    INF = 10**15
    g_score = {start_stop: start_time}
    parent = {}
    f_score = {start_stop: start_time + heuristic(start_stop)}

    open_set = [(f_score[start_stop], start_stop)]
    closed_set = set()

    while open_set:
        _, current = heappop(open_set)
        if current == end_stop:
            break
        closed_set.add(current)

        if current not in graph:
            continue
        current_g = g_score.get(current, INF)

        for edge in graph[current]:
            neighbor = edge['end_stop']
            dep = edge['dep']
            arr = edge['arr']
            line = edge['line']

            if neighbor in closed_set:
                continue

            if dep >= current_g:
                tentative_g = arr
                if tentative_g < g_score.get(neighbor, INF):
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = (current, line, arr)
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heappush(open_set, (f_score[neighbor], neighbor))

    if end_stop not in g_score:
        return [], None, None

    # Rekonstrukcja
    total_time = g_score[end_stop] - start_time
    schedule = []
    cur = end_stop
    while cur != start_stop:
        if cur not in parent:
            return [], None, None
        prev_stop, used_line, arr_time_at_cur = parent[cur]
        departure_time = g_score[prev_stop]  # bo to earliest arrival w prev_stop
        schedule.append((used_line, prev_stop, departure_time, cur, arr_time_at_cur))
        cur = prev_stop

    schedule.reverse()
    return schedule, g_score[end_stop], total_time

def print_solution(schedule, final_arr_time, total_travel):
    if not schedule:
        print("No route found.")
        return
    print("=== A* Travel Schedule (Time) ===")
    for (line, from_stop, dep_time, to_stop, arr_time) in schedule:
        print(f"Line {line}, board at {minutes_to_time_str(dep_time)} from {from_stop}, "
              f"arrive at {minutes_to_time_str(arr_time)} in {to_stop}")
    print(f"Final arrival time: {minutes_to_time_str(final_arr_time)}")
    print(f"Total travel time = {total_travel} minutes")

def main():
    if len(sys.argv) < 5:
        print("Usage: python astar_time_criteria.py <start_stop> <end_stop> <t/p> <start_time HH:MM:SS>")
        sys.exit(1)

    start_stop = sys.argv[1]
    end_stop = sys.argv[2]
    criterion = sys.argv[3]
    start_time_str = sys.argv[4]

    if criterion != 't':
        print("This script handles time-based A* only (use 't').")
        sys.exit(1)

    start_minutes = time_to_minutes(start_time_str)
    comp_start = pytime.time()

    graph, coords = parse_csv("connection_graph.csv")
    schedule, final_arr, total_travel = astar_time(graph, coords, start_stop, end_stop, start_minutes)

    comp_end = pytime.time()
    print_solution(schedule, final_arr, total_travel)
    sys.stderr.write(f"Computation time = {comp_end - comp_start:.4f} s\n")

if __name__ == "__main__":
    main()
