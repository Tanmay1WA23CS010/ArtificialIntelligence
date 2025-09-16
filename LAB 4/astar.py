import heapq

GOAL_STATE = ((1, 2, 3), (4, 5, 6), (7, 8, 0))
MOVES = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]

def manhattan_distance(state):
    distance = 0
    for r in range(3):
        for c in range(3):
            value = state[r][c]
            if value != 0:
                goal_r, goal_c = (value - 1) // 3, (value - 1) % 3
                distance += abs(r - goal_r) + abs(c - goal_c)
    return distance

def get_neighbors(state):
    neighbors = []
    for r in range(3):
        for c in range(3):
            if state[r][c] == 0:
                blank_pos = (r, c)
                break
    for move in MOVES:
        new_r, new_c = blank_pos[0] + move[0], blank_pos[1] + move[1]
        if 0 <= new_r < 3 and 0 <= new_c < 3:
            new_state = [list(row) for row in state]
            new_state[blank_pos[0]][blank_pos[1]], new_state[new_r][new_c] = new_state[new_r][new_c], new_state[blank_pos[0]][blank_pos[1]]
            neighbors.append((tuple(tuple(row) for row in new_state), move[2]))
    return neighbors

def a_star(start_state):
    open_list = []
    heapq.heappush(open_list, (0 + manhattan_distance(start_state), 0, start_state, []))
    visited = set()
    visited.add(start_state)
    
    while open_list:
        f, g, current_state, path = heapq.heappop(open_list)
        
        if current_state == GOAL_STATE:
            return path
        
        for neighbor, move_direction in get_neighbors(current_state):
            if neighbor not in visited:
                visited.add(neighbor)
                new_g = g + 1
                new_h = manhattan_distance(neighbor)
                new_f = new_g + new_h
                new_path = path + [(neighbor, move_direction)]
                heapq.heappush(open_list, (new_f, new_g, neighbor, new_path))
    
    return None

def print_state(state):
    for row in state:
        print(row)
    print()

start_state = (
    (1, 2, 3),
    (4, 0, 6),
    (7, 5, 8)
)

print("Initial State:")
print_state(start_state)

solution_path = a_star(start_state)

if solution_path:
    print("Solution found! Path to goal:")
    for step, move in solution_path:
        print(f"Move: {move}")
        print_state(step)
else:
    print("No solution found.")

print("Final Goal State:")
print_state(GOAL_STATE)
