import numpy as np
from typing import List
import random
import networkx as nx

# from littleballoffur import LoopErasedRandomWalkSampler, NonBackTrackingRandomWalkSampler

from dataclasses_rect_point import Rectangle, Point


def center_offset(rect: Rectangle) -> Point:
    """Return the offset of the center of a Rectangle from the lower left corner."""
    return Point(rect.width / 2, rect.height / 2)


def convert_center_to_lower_left(rects: List[Rectangle]) -> List[Rectangle]:
    """Convert the center of a list of Rectangles to the lower left corner."""
    for rect in rects:
        if rect.lower_left is None:
            continue
        rect.lower_left = rect.lower_left - center_offset(rect)
    return rects


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


def rectangles_overlap(rect1: Rectangle, rect2: Rectangle) -> bool:
    """Check if two rectangles overlap."""
    return (
        rect1.lower_left.x < rect2.lower_left.x + rect2.width
        and rect1.lower_left.x + rect1.width > rect2.lower_left.x
        and rect1.lower_left.y < rect2.lower_left.y + rect2.height
        and rect1.lower_left.y + rect1.height > rect2.lower_left.y
    )


def sample_graph(adj: np.ndarray) -> np.ndarray:
    """Sample random neighbours from an adjacency matrix until all nodes are visited.
    Return adjacency matrix of the sampled edges."""
    n = adj.shape[0]
    visited = [False] * n
    for i in range(n):
        if sum(adj[i,:]) == 0:
            print("Node", i, "has no neighbours")
            return adj
    visited[random.randint(0, n - 1)] = True
    graph = nx.from_numpy_array(adj)
    new_adj = np.zeros((n, n))
    while not all(visited):
        start = random.choice([i for i, v in enumerate(visited) if v])
        neighbours = list(graph.neighbors(start))
        if not neighbours:
            continue
        next_node = random.choice(neighbours)
        if visited[next_node]:
            continue
        visited[next_node] = True
        new_adj[start][next_node] = 1
        new_adj[next_node][start] = 1
    return new_adj
