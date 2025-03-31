import csv
import sys
import time
import heapq


def parse_csv(filename):
    graph = {}
    with open(filename, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = row["start_stop"]
            end = row["end_stop"]
            dep_time = row["departure_time"]
            arr_time = row["arrival_time"]
            line = row["line"]
            company = row["company"]
            dep_minutes = time_to_minutes(dep_time)
            arr_minutes = time_to_minutes(arr_time)
            graph.setdefault(start, []).append({
                'end_stop': end,
                'dep': dep_minutes,
                'arr': arr_minutes,
                'line': line,
                'company': company
            })

    return graph


def time_to_minutes(hhmmss: str) -> int:
    hh, mm, ss = hhmmss.split(':')
    return int(hh) * 60 + int(mm) + int(ss) // 60


def minutes_to_time_str(total_minutes: int) -> str:
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}:00"


def dijkstra_time(graph, start_stop, end_stop, start_time):
    INF = 10**9
    dist = {start_stop: start_time}
    parent = {}
    pq = []
    heapq.heappush(pq, (start_time, start_stop))
    while pq:
        curr_time, stop = heapq.heappop(pq)
        if stop == end_stop:
            break

        if curr_time > dist.get(stop, INF):
            continue

        if stop not in graph:
            continue

        for edge in graph[stop]:
            dep = edge['dep']
            arr = edge['arr']
            next_stop = edge['end_stop']
            if dep >= curr_time:
                if arr < dist.get(next_stop, INF):
                    dist[next_stop] = arr
                    parent[next_stop] = (stop, edge['line'], arr)
                    heapq.heappush(pq, (arr, next_stop))

    if end_stop not in dist:
        return [], None, None

    total_travel_time = dist[end_stop] - start_time
    schedule_reversed = []
    curr = end_stop
    while curr != start_stop:
        if curr not in parent:
            return [], None, None
        prev_stop, used_line, arr_time = parent[curr]
        schedule_reversed.append((used_line, prev_stop, curr, arr_time))
        curr = prev_stop

    schedule_reversed.reverse()
    final_schedule = []
    current_departure_time = start_time
    for (line, from_stop, to_stop, arrival_time) in schedule_reversed:
        final_schedule.append({
            "line": line,
            "from_stop": from_stop,
            "dep_time": current_departure_time,
            "to_stop": to_stop,
            "arr_time": arrival_time
        })
        current_departure_time = arrival_time

    return final_schedule, dist[end_stop], total_travel_time


def print_solution(schedule, arrival_time, total_time):
    if not schedule:
        print("No route found.")
        return

    print("=== Travel Schedule ===")
    for segment in schedule:
        dep_str = minutes_to_time_str(segment["dep_time"])
        arr_str = minutes_to_time_str(segment["arr_time"])
        line = segment["line"]
        print(f"Line {line}, board at {dep_str} from {segment['from_stop']}, arrive at {arr_str} in {segment['to_stop']}")

    print(f"Final arrival time: {minutes_to_time_str(arrival_time)}")
    print("=========================")
    print(f"Total travel time: {total_time} minutes")


def main():
    if len(sys.argv) < 5:
        print("Usage: python dijkstra_time_criteria.py <start> <end> <t> <start_time>")
        print("Example: python task1_dijkstra_time.py 'ZOO' 'Kwiska' t '06:00:00'")
        sys.exit(1)

    start_stop = sys.argv[1]
    end_stop = sys.argv[2]
    criterion = sys.argv[3]
    user_start_time_str = sys.argv[4]
    if criterion != 't':
        print("This script implements only the time-based Dijkstra. Use 't' as criterion.")
        sys.exit(1)

    start_time_minutes = time_to_minutes(user_start_time_str)
    comp_start = time.time()
    graph = parse_csv("connection_graph.csv")
    path, arrival_time, total_travel_time = dijkstra_time(graph, start_stop, end_stop, start_time_minutes)
    comp_end = time.time()
    comp_duration = comp_end - comp_start
    print_solution(path, arrival_time, total_travel_time)
    sys.stderr.write(f"Total travel time = {total_travel_time} minutes\n")
    sys.stderr.write(f"Computation time = {comp_duration:.4f} s\n")


if __name__ == "__main__":
    main()