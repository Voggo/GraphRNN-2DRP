from typing import List
import random
import queue
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


from plot_rects import *
from dataclasses_rect_point import Rectangle, Point
from utils import *
from generator import *


if __name__ == "__main__":
    for i in range(1):
        i = 7
        random.seed(i)

        # Generate random rectangles
        rects = generate_rects(50, 50, 5)
        # convert np.aaay to list of rectangles
        rects_list: list[Rectangle] = rects.flatten().tolist()
        rects_list = list(set(rects_list))

        # Plot the rectangles
        plot_rects(rects_list, 50, 50, show_number=False)

        # Reduce the number of rectangles
        reduced_rects = reduce_rects(rects, convergence_limit=100)
        # Plot the rectangles
        if len(reduced_rects) < 3:
            continue
        plot_rects(reduced_rects, 50, 50, show_number=True)

        # Convert the rectangles to a graph
        adj, edir, offset = convert_rects_to_graph(reduced_rects)
        # Plot the graph

        nodes = []
        for rect in reduced_rects:
            nodes.append(copy.copy(rect))
        for node in nodes:
            node.lower_left = None

        bfs_order = bfs_index(adj, 0)
        bfs_nodes = [nodes[i] for i in bfs_order]
        bfs_adj = adj[np.ix_(bfs_order, bfs_order)]
        bfs_edge_dir = edir[np.ix_(bfs_order, bfs_order)]
        bfs_offset = offset[np.ix_(bfs_order, bfs_order)]

        if len(bfs_nodes) < 3:
            continue
        show_graph_with_labels(bfs_adj, {i: i for i in range(len(bfs_nodes))})          
        print(i)