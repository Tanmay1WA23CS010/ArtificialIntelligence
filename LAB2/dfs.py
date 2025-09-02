# 8-Puzzle using DFS with move directions

goal_state = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]]  # 0 is the blank tile

# Directions: (dx, dy, move_name)
moves = [(-1, 0, "Up"),
         (1, 0, "Down"),
         (0, -1, "Left"),
         (0, 1, "Right")]


def find_blank(state):
    """Find the blank (0) position."""
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j


def is_goal(state):
    return state == goal_state


def print_state(state):
    for row in state:
        print(row)
    print()


def dfs(state, visited, path, depth=0, max_depth=20):
    """DFS recursive search with moves."""
    if is_goal(state):
        print("✅ Goal reached in", depth, "moves")
        print("Moves:", " -> ".join(path))
        print()
        return True

    if depth >= max_depth:
        return False

    visited.add(tuple(map(tuple, state)))

    x, y = find_blank(state)

    for dx, dy, move in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            # Copy state
            new_state = [row[:] for row in state]
            # Swap blank
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]

            if tuple(map(tuple, new_state)) not in visited:
                if dfs(new_state, visited, path + [move], depth + 1, max_depth):
                    return True
    return False


# Example
initial_state = [[1, 2, 3],
                 [4, 0, 6],
                 [7, 5, 8]]

print("Initial State:")
print_state(initial_state)

visited = set()
if not dfs(initial_state, visited, []):
    print("❌ Goal not found within depth limit.")
