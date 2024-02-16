import random
import numpy as np
from typing import List

from plot_rects import plot_rects
from dataclasses_rect_point import Rectangle, Point


def generate_rects(width: int, height: int, n_breaks: int) -> np.ndarray:
    """Generate a list of Rectangles with n_breaks line breaks in the x and y directions."""
    if width <= 0 or height <= 0 or n_breaks <= 0:
        raise ValueError("All parameters must be positive")
    x_segments = get_line_breaks(width, n_breaks)  # Add 0 to the beginning of the list
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

def rectangles_overlap(rect1: Rectangle, rect2: Rectangle) -> bool:
    """Check if two rectangles overlap."""
    return (
        rect1.lower_left.x < rect2.lower_left.x + rect2.width
        and rect1.lower_left.x + rect1.width > rect2.lower_left.x
        and rect1.lower_left.y < rect2.lower_left.y + rect2.height
        and rect1.lower_left.y + rect1.height > rect2.lower_left.y
    )

def reduce_rects(rects: np.ndarray, n_rects: int) -> List[Rectangle]:
    """Return a list of n_rects random Rectangles from a list of Rectangles."""
    if n_rects >= rects.size:
        return rects.flatten().tolist()

    cap = 0  # Temporary cap to prevent infinite loop
    while np.count_nonzero(rects != None) > n_rects:
        x = random.randint(0, rects.shape[0] - 1)
        y = random.randint(0, rects.shape[1] - 1)
        direction = random.choice(([1, 0], [0, 1], [-1, 0], [0, -1]))
        x_compare = x + direction[0]
        y_compare = y + direction[1]

        # Check if the comparison indices are within bounds
        if (
            x_compare >= rects.shape[0]
            or y_compare >= rects.shape[1]
            or x_compare < 0
            or y_compare < 0
        ):
            continue

        # Check if both rectangles are not None and there is no overlap
        if (
            rects[x, y] is not None
            and rects[x_compare, y_compare] is not None
            and not rectangles_overlap(rects[x, y], rects[x_compare, y_compare])
        ):
            # Merge the rectangles
            direction = [x_compare - x, y_compare - y]
            if direction[0] > 0:
                if rects[x, y].height == rects[x_compare, y_compare].height:
                    rects[x, y].width += rects[x_compare, y_compare].width
                    rects[x_compare, y_compare] = None
            if direction[1] > 0:
                if rects[x, y].width == rects[x_compare, y_compare].width:
                    rects[x, y].height += rects[x_compare, y_compare].height
                    rects[x_compare, y_compare] = None
            if direction[0] < 0:
                if rects[x, y].height == rects[x_compare, y_compare].height:
                    rects[x_compare, y_compare].width += rects[x, y].width
                    rects[x, y] = None
            if direction[1] < 0:
                if rects[x, y].width == rects[x_compare, y_compare].width:
                    rects[x_compare, y_compare].height += rects[x, y].height
                    rects[x, y] = None

        cap += 1
        if cap > 1000:
            break

    rects = rects.flatten().tolist()
    return [rect for rect in rects if rect is not None]


def get_line_breaks(line: int, n_breaks: int) -> List[int]:
    """Return a list of n_breaks random line breaks."""
    if line <= 1 or n_breaks == 0:
        return []
    result = []
    for _ in range(n_breaks):
        while True:
            randnum = random.randint(1, line)
            if randnum not in result:
                result.append(randnum)
                break
    result.sort()
    return result


if __name__ == "__main__":
    # test_rects = generate_rects(5, 5, 3)
    # reduced_rects = reduce_rects(test_rects, 10)
    # print(reduced_rects)
    # plot_rects(reduced_rects, ax_lim=5, ay_lim=5)
    test_rects = generate_rects(50, 50, 5)
    # plot_rects(test_rects.flatten().tolist(), ax_lim=100, ay_lim=100)
    reduced_rects = reduce_rects(test_rects, 25)
    plot_rects(reduced_rects, ax_lim=50, ay_lim=50)
