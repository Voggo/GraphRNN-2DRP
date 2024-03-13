from typing import List
import random
import numpy as np
import queue

from plot_rects import plot_rects
from dataclasses_rect_point import Rectangle, Point


def generate_rects(width: int, height: int, n_breaks: int) -> np.ndarray:
    """Generate a list of Rectangles with n_breaks line breaks in the x and y directions."""
    if width <= 0 or height <= 0 or n_breaks <= 0:
        raise ValueError("All parameters must be positive")
    x_segments = get_line_breaks(width, n_breaks)
    y_segments = get_line_breaks(height, n_breaks)
    rectangles = np.empty((len(y_segments) + 1, len(x_segments) + 1), dtype=Rectangle)
    x_axis = 0
    y_axis = 0
    for y1, y2 in zip([0] + y_segments, y_segments + [height]):
        for x1, x2 in zip([0] + x_segments, x_segments + [width]):
            rectangles[x_axis, y_axis] = Rectangle(
                x2 - x1, y2 - y1, lower_left=Point(x1, y1)
            )
            x_axis += 1
        y_axis += 1
        x_axis = 0
    return rectangles


# might not be needed anymore
def rectangles_overlap(rect1: Rectangle, rect2: Rectangle) -> bool:
    """Check if two rectangles overlap."""
    return (
        rect1.lower_left.x < rect2.lower_left.x + rect2.width
        and rect1.lower_left.x + rect1.width > rect2.lower_left.x
        and rect1.lower_left.y < rect2.lower_left.y + rect2.height
        and rect1.lower_left.y + rect1.height > rect2.lower_left.y
    )


def reduce_rects(rects: np.ndarray, convergence_limit=1000) -> List[Rectangle]:
    """Return a list of randomly reduced Rectangles from a list of Rectangles."""

    convergence = 0
    while True:
        x = random.randint(0, rects.shape[0] - 1)
        y = random.randint(0, rects.shape[1] - 1)
        direction = random.choice(([1, 0], [0, 1], [-1, 0], [0, -1]))
        x_compare = x
        y_compare = y

        while rects[x, y].lower_left == rects[x_compare, y_compare].lower_left:
            if (
                direction[0] > 0
                and x_compare + 1 >= rects.shape[0]
                or direction[0] < 0
                and x_compare - 1 < 0
                or direction[1] > 0
                and y_compare + 1 >= rects.shape[1]
                or direction[1] < 0
                and y_compare - 1 < 0
            ):
                break
            else:
                x_compare += direction[0]
                y_compare += direction[1]

        # Check if the comparison indices are within bounds
        if x == x_compare and y == y_compare:
            continue

        # Check if both rectangles are not None and there is no overlap
        if (
            rects[x, y] is not None
            and rects[x_compare, y_compare] is not None
            and not rectangles_overlap(rects[x, y], rects[x_compare, y_compare])
        ):
            # Merge the rectangles
            if direction[0] > 0:
                if (
                    rects[x, y].height == rects[x_compare, y_compare].height
                    and (rects[x, y].lower_left.x + rects[x, y].width)
                    == rects[x_compare, y_compare].lower_left.x
                    and rects[x, y].lower_left.y
                    == rects[x_compare, y_compare].lower_left.y
                ):
                    index = np.where(rects == rects[x_compare, y_compare])
                    rects[x, y].width += rects[x_compare, y_compare].width
                    rects[index] = rects[x, y]
                    convergence = 0
            if direction[1] > 0:
                if (
                    rects[x, y].width == rects[x_compare, y_compare].width
                    and (rects[x, y].lower_left.y + rects[x, y].height)
                    == rects[x_compare, y_compare].lower_left.y
                    and rects[x, y].lower_left.x
                    == rects[x_compare, y_compare].lower_left.x
                ):
                    index = np.where(rects == rects[x_compare, y_compare])
                    rects[x, y].height += rects[x_compare, y_compare].height
                    rects[index] = rects[x, y]
                    convergence = 0
            if direction[0] < 0:
                if (
                    rects[x, y].height == rects[x_compare, y_compare].height
                    and (rects[x_compare, y_compare].lower_left.x - rects[x, y].width)
                    == rects[x, y].lower_left.x
                    and rects[x, y].lower_left.y
                    == rects[x_compare, y_compare].lower_left.y
                ):
                    index = np.where(rects == rects[x, y])
                    rects[x_compare, y_compare].width += rects[x, y].width
                    rects[index] = rects[x_compare, y_compare]
                    convergence = 0
            if direction[1] < 0:
                if (
                    rects[x, y].width == rects[x_compare, y_compare].width
                    and rects[x_compare, y_compare].lower_left.y - rects[x, y].height
                    == rects[x, y].lower_left.y
                    and rects[x, y].lower_left.x
                    == rects[x_compare, y_compare].lower_left.x
                ):
                    index = np.where(rects == rects[x, y])
                    rects[x_compare, y_compare].height += rects[x, y].height
                    rects[index] = rects[x_compare, y_compare]
                    convergence = 0

        convergence += 1
        if convergence > convergence_limit:
            print("Convergence limit reached. Exiting...")
            break

    rects_list: list[Rectangle] = rects.flatten().tolist()
    rects_list = list(set(rects_list))
    return [rect for rect in rects_list if rect is not None]


def convert_rects_to_graph(
    rects: List[Rectangle],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of Rectangles to a graph."""
    adjacency_matrix = np.zeros((len(rects), len(rects)), dtype=int)
    edge_directions = np.zeros((len(rects), len(rects)), dtype=tuple)
    edge_angle = np.zeros((len(rects), len(rects)), dtype=float)
    for i, rect1 in enumerate(rects):
        for j, rect2 in enumerate(rects):
            if i == j:
                continue
            if (
                rect1.lower_left.x + rect1.width == rect2.lower_left.x
                and rect1.lower_left.y <= rect2.lower_left.y + rect2.height
                and rect1.lower_left.y + rect1.height >= rect2.lower_left.y
            ):
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
                edge_directions[i, j] = (2, 4)
                edge_directions[j, i] = (4, 2)
                rect1_center = rect1.lower_left + Point(
                    rect1.width / 2, rect1.height / 2
                )
                rect2_center = rect2.lower_left + Point(
                    rect2.width / 2, rect2.height / 2
                )
                angle = np.arctan2(
                    rect2_center.y - rect1_center.y, rect2_center.x - rect1_center.x
                )
                edge_angle[i, j] = angle
                edge_angle[j, i] = angle + np.pi
            if (
                rect1.lower_left.y + rect1.height == rect2.lower_left.y
                and rect1.lower_left.x <= rect2.lower_left.x + rect2.width
                and rect1.lower_left.x + rect1.width >= rect2.lower_left.x
            ):
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
                edge_directions[i, j] = (1, 3)
                edge_directions[j, i] = (1, 3)
                rect1_center = rect1.lower_left + Point(
                    rect1.width / 2, rect1.height / 2
                )
                rect2_center = rect2.lower_left + Point(
                    rect2.width / 2, rect2.height / 2
                )
                angle = np.arctan2(
                    rect2_center.x - rect1_center.x, rect2_center.y - rect1_center.y
                )
    return adjacency_matrix, edge_directions, edge_angle


def convert_graph_to_rects(nodes, adj, edge_dir, edge_ang):
    """Convert a graph to a list of Rectangles."""
    edge_index = np.where(adj == 1)
    queue_index = edge_index[0][0]
    visited = np.zeros(len(nodes), dtype=bool)
    q = queue.Queue()
    nodes[queue_index].lower_left = Point(0, 0)
    q.put(edge_index[1][0])

    while not q.empty():
        node_from = q.get()
        if visited[node_from]:
            continue
        visited[node_from] = True
        queue_index = np.where(edge_index[0] == node_from)
        neighbours = edge_index[1][queue_index]
        for node_to in neighbours:
            q.put(edge_index[1][node_to])
            if edge_dir[node_from, node_to] == (2, 4):
                x_offset = (nodes[node_from].width + nodes[node_from].width) / 2
                y_offset = (nodes[node_from].height / 2) + (
                    x_offset / np.tan(edge_ang[node_from, node_to])
                )
                nodes[node_to].lower_left = nodes[node_from].lower_left + Point(
                    x_offset, y_offset
                )
            if edge_dir[node_from, node_to] == (4, 2):
                x_offset = (nodes[node_from].width + nodes[node_from].width) / 2
                y_offset = (nodes[node_from].height / 2) + (
                    x_offset / np.tan(edge_ang[node_from, node_to] - np.pi)
                )
                nodes[node_to].lower_left = nodes[node_from].lower_left + Point(
                    x_offset, y_offset
                )
            if edge_dir[node_from, node_to] == (1, 3):
                pass
            if edge_dir[node_from, node_to] == (3, 1):
                pass
    return nodes


def print_rects(rects: np.ndarray) -> None:
    """Print the list of rectangles."""
    for rect in rects:
        print(rect)


def get_line_breaks(line: int, n_breaks: int) -> List[int]:
    """Return a list of n_breaks random line breaks."""
    if line <= 1 or n_breaks == 0:
        return []
    result = []
    for _ in range(n_breaks):
        while True:
            randnum = random.randint(1, line - 1)
            if randnum not in result:
                result.append(randnum)
                break
    result.sort()
    return result


if __name__ == "__main__":
    # for i in range(1):
    #     test_rects = generate_rects(50, 50, 10)
    #     reduced_rects = reduce_rects(test_rects, convergence_limit=100)
    #     plot_rects(
    #         reduced_rects, ax_lim=50, ay_lim=50, filename=f"test_{i}.png", show=False
    #     )
    #     print(f"Test {i}: {len(reduced_rects)} rectangles")

    test_rects = generate_rects(50, 50, 10)
    reduced_rects = reduce_rects(test_rects, convergence_limit=100)
    plot_rects(
        reduced_rects, ax_lim=50, ay_lim=50, filename=f"test_graph.png", show=True
    )
    print(f"Test: {len(reduced_rects)} rectangles")
    adjacency_matrix, edge_directions, edge_angle = convert_rects_to_graph(
        reduced_rects
    )
    print(adjacency_matrix)
    print(edge_directions)
    index = np.where(edge_angle != 0)
    # for i, j in zip(index[0], index[1]):
    #     print(f"Angle between {i} and {j}: {edge_angle[i, j]}")
    #     print(f"Angle between {j} and {i}: {edge_angle[j, i]}")
    #     print(edge_angle[i, j] - edge_angle[j, i])
    nodes = reduced_rects
    for node in nodes:
        node.lower_left = None
    rects_again = convert_graph_to_rects(
        nodes, adjacency_matrix, edge_directions, edge_angle
    )
    plot_rects(
        rects_again,
        ax_lim=60,
        ay_lim=60,
        filename=f"test_graph_to_rects.png",
        show=True,
    )
