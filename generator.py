import random
import numpy as np
from typing import List

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


def rectangles_overlap(rect1: Rectangle, rect2: Rectangle) -> bool: # might not be needed anymore
    """Check if two rectangles overlap."""
    if rect1 is None or rect2 is None:
        return False  # Return False if either rectangle is None
    return (
        rect1.lower_left.x < rect2.lower_left.x + rect2.width
        and rect1.lower_left.x + rect1.width > rect2.lower_left.x
        and rect1.lower_left.y < rect2.lower_left.y + rect2.height
        and rect1.lower_left.y + rect1.height > rect2.lower_left.y
    )


def reduce_rects(rects: np.ndarray, convergence_limit=1000) -> List[Rectangle]:
    """Return a list of randomly reduced Rectangles from a list of Rectangles."""

    convergence = 0  # Temporary cap to prevent infinite loop
    while True:
        x = random.randint(0, rects.shape[0] - 1)
        y = random.randint(0, rects.shape[1] - 1)
        direction = random.choice(([1, 0], [0, 1], [-1, 0], [0, -1])) # 
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
            # and not rectangles_overlap(rects[x, y], rects[x_compare, y_compare])
        ):
            # Merge the rectangles
            direction = [x_compare - x, y_compare - y]
            if direction[0] > 0:
                if (
                    rects[x, y].height == rects[x_compare, y_compare].height
                    and rects[x, y].lower_left != rects[x_compare, y_compare].lower_left
                ):
                    index = np.where(rects == rects[x_compare, y_compare])
                    rects[x, y].width += rects[x_compare, y_compare].width
                    # rects[x_compare, y_compare] = rects[x, y]
                    rects[index] = rects[x, y]
                    print("Rectangle removed")
            if direction[1] > 0:
                if rects[x, y].width == rects[x_compare, y_compare].width and rects[x, y].lower_left != rects[x_compare, y_compare].lower_left:
                    index = np.where(rects == rects[x_compare, y_compare])
                    rects[x, y].height += rects[x_compare, y_compare].height
                    rects[index] = rects[x, y]
                    print("Rectangle removed")
                    convergence = 0
            if direction[0] < 0:
                if rects[x, y].height == rects[x_compare, y_compare].height and rects[x, y].lower_left != rects[x_compare, y_compare].lower_left:
                    index = np.where(rects == rects[x, y])
                    rects[x_compare, y_compare].width += rects[x, y].width
                    rects[index] = rects[x_compare, y_compare]
                    print("Rectangle removed")
                    convergence = 0
            if direction[1] < 0:
                if rects[x, y].width == rects[x_compare, y_compare].width and rects[x, y].lower_left != rects[x_compare, y_compare].lower_left:
                    index = np.where(rects == rects[x, y])
                    rects[x_compare, y_compare].height += rects[x, y].height
                    rects[index] = rects[x_compare, y_compare]
                    print("Rectangle removed")
                    convergence = 0
        # print(f"{convergence}: x: {x}, y: {y}, x_compare: {x_compare}, y_compare: {y_compare}")
    
        convergence += 1
        if convergence > convergence_limit:
            print("Convergence limit reached. Exiting...")
            break

    rects = rects.flatten().tolist()
    return [rect for rect in rects if rect is not None]

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
    for i in range(20):
        test_rects = generate_rects(50, 50, 10)
        reduced_rects = reduce_rects(test_rects, convergence_limit=100)
        plot_rects(reduced_rects, ax_lim=50, ay_lim=50, filename=f"test_{i}.png", show=False)
