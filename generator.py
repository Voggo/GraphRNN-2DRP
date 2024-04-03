from typing import List
import random
import queue
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from plot_rects import plot_rects
from dataclasses_rect_point import Rectangle, Point
from utils import *

# random.seed(5)
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


def reduce_rects(rects: np.ndarray, convergence_limit=100) -> List[Rectangle]:
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
    for i, rect1_enum in enumerate(rects):
        for j, rect2_enum in enumerate(rects):
            index_i, index_j = i, j
            rect1, rect2 = copy.copy(rect1_enum), copy.copy(rect2_enum)
            if i == j:
                continue
            if (
                (
                    rect1.lower_left.x + rect1.width == rect2.lower_left.x
                    or rect1.lower_left.x - rect2.width == rect2.lower_left.x
                )
                and rect1.lower_left.y < rect2.lower_left.y + rect2.height
                and rect1.lower_left.y + rect1.height > rect2.lower_left.y
            ):
                if rect1.lower_left.x - rect2.width == rect2.lower_left.x:
                    index_i, index_j = index_j, index_i
                    rect1, rect2 = copy.copy(rect2), copy.copy(rect1)
                adjacency_matrix[index_i, index_j] = 1
                adjacency_matrix[index_j, index_i] = 1
                edge_directions[index_i, index_j] = (2, 4)
                edge_directions[index_j, index_i] = (4, 2)
                rect1_center = rect1.lower_left + center_offset(rect1)
                rect2_center = rect2.lower_left + center_offset(rect2)
                angle = np.arctan2(
                    rect2_center.y - rect1_center.y, rect2_center.x - rect1_center.x
                )
                edge_angle[index_i, index_j] = angle
                edge_angle[index_j, index_i] = angle
            elif (
                (
                    rect1.lower_left.y + rect1.height == rect2.lower_left.y
                    or rect1.lower_left.y - rect2.height == rect2.lower_left.y
                )
                and rect1.lower_left.x < rect2.lower_left.x + rect2.width
                and rect1.lower_left.x + rect1.width > rect2.lower_left.x
            ):
                if rect1.lower_left.y - rect2.height == rect2.lower_left.y:
                    index_i, index_j = index_j, index_i
                    rect1, rect2 = copy.copy(rect2), copy.copy(rect1)
                adjacency_matrix[index_i, index_j] = 1
                adjacency_matrix[index_j, index_i] = 1
                edge_directions[index_i, index_j] = (1, 3)
                edge_directions[index_j, index_i] = (3, 1)
                rect1_center = rect1.lower_left + center_offset(rect1)
                rect2_center = rect2.lower_left + center_offset(rect2)
                angle = np.arctan2(
                    rect2_center.x - rect1_center.x, rect2_center.y - rect1_center.y
                )
                edge_angle[index_i, index_j] = angle
                edge_angle[index_j, index_i] = angle
    return adjacency_matrix, edge_directions, edge_angle


def convert_graph_to_rects(nodes, adj, edge_dir, edge_ang):
    """Convert a graph to a list of Rectangles."""
    print(adj)
    edge_index = np.where(adj == 1)
    queue_index = edge_index[0][0]
    visited = np.zeros(len(nodes), dtype=bool)
    q = queue.Queue()
    nodes[queue_index].lower_left = Point(0, 0)
    q.put(edge_index[0][0])
    count = 0
    while not q.empty():
        node_from = q.get()
        visited[node_from] = True
        queue_index = np.where(edge_index[0] == node_from)
        neighbours = edge_index[1][queue_index]
        for node_to in neighbours:
            if visited[node_to] or nodes[node_to].lower_left is not None:
                continue
            count += 1
            q.put(node_to)
            if edge_dir[node_from, node_to] == (2, 4):
                x_offset = (nodes[node_from].width + nodes[node_to].width) / 2
                y_offset = (
                    (x_offset * np.tan(edge_ang[node_from, node_to]))
                    if not np.isclose(edge_ang[node_from, node_to], 0.0, 1e-2)
                    else 0.0
                )
                nodes[node_to].lower_left = nodes[node_from].lower_left + Point(
                    x_offset, y_offset
                )
            elif edge_dir[node_from, node_to] == (4, 2):
                x_offset = (nodes[node_from].width + nodes[node_to].width) / 2
                y_offset = (
                    (x_offset * np.tan(edge_ang[node_from, node_to]))
                    if not np.isclose(edge_ang[node_from, node_to], 0.0, 1e-2)
                    else 0.0
                )
                nodes[node_to].lower_left = nodes[node_from].lower_left + Point(
                    -1 * x_offset, -1 * y_offset
                )
            elif edge_dir[node_from, node_to] == (1, 3):
                y_offset = (nodes[node_from].height + nodes[node_to].height) / 2
                x_offset = (
                    y_offset * np.tan(edge_ang[node_from, node_to])
                    if not np.isclose(edge_ang[node_from, node_to], 0.0, 1e-2)
                    else 0.0
                )
                nodes[node_to].lower_left = nodes[node_from].lower_left + Point(
                    x_offset, y_offset
                )
            elif edge_dir[node_from, node_to] == (3, 1):
                y_offset = (nodes[node_from].height + nodes[node_to].height) / 2
                x_offset = (
                    y_offset * np.tan(edge_ang[node_from, node_to])
                    if not np.isclose(edge_ang[node_from, node_to], 0.0, 1e-2)
                    else 0.0
                )
                nodes[node_to].lower_left = nodes[node_from].lower_left + Point(
                    -1 * x_offset, -1 * y_offset
                )
    return nodes


def generate_rects_and_graph(
    width: int, height: int, n_breaks: int, convergence_limit=100
) -> tuple:
    """Generate a list of Rectangles and convert it to a graph."""
    rects = generate_rects(width, height, n_breaks)
    reduced_rects = reduce_rects(rects, convergence_limit=100)
    if len(reduced_rects) == 1:
        rects = generate_rects(width, height, n_breaks)
        reduced_rects = reduce_rects(rects, convergence_limit=100)
    adjacency_matrix, edge_directions, edge_angle = convert_rects_to_graph(
        reduced_rects
    )
    return reduced_rects, adjacency_matrix, edge_directions, edge_angle


def show_graph_with_labels(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    plt.savefig("plots_img/graph", )
    plt.show()


if __name__ == "__main__":
    (
        reduced_rects,
        adjacency_matrix,
        edge_directions,
        edge_angle,
    ) = generate_rects_and_graph(50, 50, 6)
    plot_rects(
        reduced_rects, ax_lim=50, ay_lim=50, filename="test_graph.png", show=False
    )
    nodes = []
    for rect in reduced_rects:
        nodes.append(copy.copy(rect))
    for node in nodes:
        node.lower_left = None
    rects_again = convert_graph_to_rects(
        nodes, adjacency_matrix, edge_directions, edge_angle
    )
    rects_again = convert_center_to_lower_left(rects_again)
    plot_rects(
        rects_again,
        ax_min=-50,
        ay_min=-50,
        ax_lim=50,
        ay_lim=50,
        filename="test_graph_to_rects.png",
        show=True,
    )
    show_graph_with_labels(adjacency_matrix, {i: i for i in range(len(reduced_rects))})