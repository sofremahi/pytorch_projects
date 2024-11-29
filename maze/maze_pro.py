import sys
from PIL import Image, ImageDraw


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action


class Frontier:
    def __init__(self, use_stack=True):
        self.frontier = []
        self.use_stack = use_stack

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("Frontier is empty")
        return self.frontier.pop() if self.use_stack else self.frontier.pop(0)


class Maze:
    def __init__(self, filename):
        # Read maze file
        with open(filename) as f:
            contents = f.read()

        # Validate start and goal
        if contents.count("A") != 1 or contents.count("B") != 1:
            raise ValueError("Maze must contain exactly one start (A) and one goal (B)")

        # Parse maze structure
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(row) for row in contents)
        self.start, self.goal = None, None
        self.walls = []

        for i, row in enumerate(contents):
            row_data = []
            for j, char in enumerate(row.ljust(self.width)):
                if char == "A":
                    self.start = (i, j)
                    row_data.append(False)
                elif char == "B":
                    self.goal = (i, j)
                    row_data.append(False)
                elif char == " ":
                    row_data.append(False)
                else:
                    row_data.append(True)
            self.walls.append(row_data)

        if not self.start or not self.goal:
            raise ValueError("Maze must have valid start (A) and goal (B) points")

        self.solution = None
        self.num_explored = 0
        self.explored = set()

    def print(self):
        solution = self.solution[1] if self.solution else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()

    def neighbors(self, state):
        row, col = state
        directions = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1)),
        ]
        return [
            (action, (r, c))
            for action, (r, c) in directions
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]
        ]

    def solve(self):
        self.num_explored = 0
        start_node = Node(state=self.start)
        frontier = Frontier(use_stack=True)
        frontier.add(start_node)

        while not frontier.empty():
            current_node = frontier.remove()
            self.num_explored += 1

            if current_node.state == self.goal:
                actions, cells = [], []
                while current_node.parent is not None:
                    actions.append(current_node.action)
                    cells.append(current_node.state)
                    current_node = current_node.parent
                self.solution = (list(reversed(actions)), list(reversed(cells)))
                return

            self.explored.add(current_node.state)

            for action, state in self.neighbors(current_node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child_node = Node(state=state, parent=current_node, action=action)
                    frontier.add(child_node)

        raise Exception("No solution")

    def output_image(self, filename, show_solution=True, show_explored=False):
        cell_size = 50
        cell_border = 2
        img = Image.new("RGBA", (self.width * cell_size, self.height * cell_size), "black")
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    fill = (40, 40, 40)
                elif (i, j) == self.start:
                    fill = (255, 0, 0)
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)
                elif solution and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)
                elif show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)
                else:
                    fill = (237, 240, 252)

                draw.rectangle(
                    [
                        (j * cell_size + cell_border, i * cell_size + cell_border),
                        ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border),
                    ],
                    fill=fill,
                )
        img.save(filename)


if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze.txt")

maze = Maze(sys.argv[1])
print("Maze:")
maze.print()
print("Solving...")
maze.solve()
print(f"States Explored: {maze.num_explored}")
print("Solution:")
maze.print()
maze.output_image("maze_solution.png", show_explored=True)
