from random import randint
from typing import List
import numpy as np

from plot_rects import plot_rects
from dataclasses_rect_point import Rectangle, Point


def generate_rects(width: int, height: int, n_breaks=10) -> np.ndarray:
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


def reduce_rects(rects: np.ndarray, n_rects: int) -> List[Rectangle]:
    """Return a list of n_rects random Rectangles from a list of Rectangles."""
    if n_rects >= rects.size:
        return rects.flatten().tolist()

    cap = 0
    while np.count_nonzero(rects != None) > n_rects:
        x = randint(0, rects.shape[0] - 1)
        y = randint(0, rects.shape[1] - 1)
        i = 0
        is_none = {"x_plus": False, "y_plus": False, "x_minus": False, "y_minus": False}
        while True:
            i += 1
            if rects[x, y] is None:
                break

            if (
                is_none["x_plus"] is False
                and x + i < rects.shape[0]
                and rects[x + 1, y] is not None
                and rects[x, y].height == rects[x + 1, y].height
            ):
                rects[x, y].width = rects[x, y].width + rects[x + 1, y].width
                rects[x + 1, y] = None
                break
            else:
                is_none["x_plus"] = True
            if (
                is_none["y_plus"] is False
                and y + i < rects.shape[1]
                and rects[x, y + 1] is not None
                and rects[x, y].width == rects[x, y + 1].width
            ):
                rects[x, y].height = rects[x, y].height + rects[x, y + 1].height
                rects[x, y + 1] = None
                break
            else:
                is_none["y_plus"] = True
            if (
                is_none["x_minus"] is False
                and x - i >= 0
                and rects[x - i, y] is not None
                and rects[x, y].height == rects[x - i, y].height
            ):
                rects[x - i, y].width = rects[x - i, y].width + rects[x, y].width
                rects[x, y] = None
                break
            else:
                is_none["x_minus"] = True
            if (
                is_none["y_minus"] is False
                and y - i >= 0
                and rects[x, y - i] is not None
                and rects[x, y].width == rects[x, y - i].width
            ):
                rects[x, y - i].height = rects[x, y - i].height + rects[x, y].height
                rects[x, y] = None
                break
            else:
                is_none["y_minus"] = True

            if (
                is_none["x_plus"]
                and is_none["y_plus"]
                and is_none["x_minus"]
                and is_none["y_minus"]
            ):
                break
        cap += 1
        if cap > 1000000:
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
            randnum = randint(1, line)
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
    reduced_rects = reduce_rects(test_rects, 5)
    plot_rects(reduced_rects, ax_lim=50, ay_lim=50)
