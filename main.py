import pandas as pd
import numpy as np

from generator import *

N_GRAPHS = 100

if __name__ == "__main__":
    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=["nodes", "adjacency", "edge_dir", "edge_angle"])

    # Display the template DataFrame
    print(df)

    for i in range(N_GRAPHS):

        rects = generate_rects(50, 50, 7)
        reduced_rects = reduce_rects(rects, convergence_limit=100)
        adjacency_matrix, edge_directions, edge_angle = convert_rects_to_graph(
            reduced_rects
        )
        df = df.append(
            {
                "nodes": reduced_rects,
                "adjacency": adjacency_matrix,
                "edge_dir": edge_directions,
                "edge_angle": edge_angle,
            }
        )
    print(df)
