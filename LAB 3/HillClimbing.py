import copy

goal_state = [[1, 2, 3],
              [8, 0, 4],
              [7, 6, 5]]

def heuristic(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            if value != 0:
                for x in range(3):
                    for y in range(3):
                        if goal_state[x][y] == value:
                            distance += abs(i - x) + abs(j - y)
    return distance

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def get_neighbors(state):
    neighbors = []
    x, y = find_blank(state)
    moves = {
        "Down": (1, 0),
        "Up": (-1, 0),
        "Right": (0, 1),
        "Left": (0, -1)
    }
    for move, (dx, dy) in moves.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = copy.deepcopy(state)
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append((move, new_state))
    return neighbors

def hill_climbing(initial_state):
    current = initial_state
    current_h = heuristic(current)
    steps = 0
    path_moves = ["Start"]

    print("\nInitial State (h={}):".format(current_h))
    for row in current:
        print(row)

    while True:
        neighbors = get_neighbors(current)

        if not neighbors:
            print("\nNo more neighbors. Stopping.")
            return current, steps, current_h, path_moves

        print("\nStep {}: Generating neighbor states...".format(steps+1))
        for move, child in neighbors:
            print(f"\nMove: {move} (h={heuristic(child)})")
            for row in child:
                print(row)

        neighbor_move, neighbor = min(neighbors, key=lambda x: heuristic(x[1]))
        neighbor_h = heuristic(neighbor)

        print("\nChosen move: {} → h={}".format(neighbor_move, neighbor_h))
        for row in neighbor:
            print(row)

        if neighbor_h >= current_h:
            print("\nNo better neighbor found. Stopping.")
            return current, steps, current_h, path_moves

        current, current_h = neighbor, neighbor_h
        steps += 1
        path_moves.append(neighbor_move)

        if current_h == 0:
            print("\nGoal Reached!")
            return current, steps, current_h, path_moves

if __name__ == "__main__":
    initial_state = [[2, 8, 3],
                     [1, 0, 4],
                     [7, 6, 5]]

    solution, steps, h_val, path_moves = hill_climbing(initial_state)

    print("\nFinal State:")
    for row in solution:
        print(row)
    print(f"Steps taken: {steps}")
    print(f"Final Heuristic value: {h_val}")

    print("\nFinal Path of Moves:")
    print(" → ".join(path_moves))