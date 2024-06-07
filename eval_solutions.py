from shapely.geometry import Polygon
from shapely import union, union_all, box, intersection

from dataclasses_rect_point import Rectangle, Point
from typing import List


def get_shapely_rects(rects: List[Rectangle]):
    """Return a list of shapely Polygon objects from a list of Rectangle dataclasses."""
    shapely_rects = []
    for rect in rects:
        try:
            if rect.lower_left is None:
                raise ValueError(
                    f"Rectangle ({rect}) did not have a lower_left attribute."
                )
            shapely_rect = box(
                rect.lower_left.x,
                rect.lower_left.y,
                rect.lower_left.x + rect.width,
                rect.lower_left.y + rect.height,
            )
            shapely_rects.append(shapely_rect)
        except ValueError as e:
            print(e)
    return shapely_rects


def get_min_max_xy(rects: List[Polygon]):
    """Return the minimum and maximum x and y coordinates from a list of shapely Polygon objects."""
    min_x = min([rect.bounds[0] for rect in rects])
    min_y = min([rect.bounds[1] for rect in rects])
    max_x = max([rect.bounds[2] for rect in rects])
    max_y = max([rect.bounds[3] for rect in rects])
    return min_x, min_y, max_x, max_y


def get_overlap_area(rects: List[Polygon]):
    intersection_area = 0
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            intersection_area += rects[i].intersection(rects[j]).area
    return intersection_area


def evaluate_solution(rects: List[Rectangle], target_width: int, target_height: int):
    """
    Evaluate the solution by calculating the area of the union of the rectangles.
    then intersecting the union with the target rectangle aligned with the
    minimum/maximum x and y coordinates, from all sides, and calculating the
    max fill ratio of the target rectangle.
    """
    shapely_rects = get_shapely_rects(rects)
    union_rects = union_all(shapely_rects)
    rects_min_x, rects_min_y, rects_max_x, rects_max_y = get_min_max_xy(shapely_rects)
    x_min_y_min_target = box(
        rects_min_x,
        rects_min_y,
        rects_min_x + target_width,
        rects_min_y + target_height,
    )
    x_max_y_max_target = box(
        rects_max_x - target_width,
        rects_max_y - target_height,
        rects_max_x,
        rects_max_y,
    )
    x_min_y_max_target = box(
        rects_min_x,
        rects_max_y - target_height,
        rects_min_x + target_width,
        rects_max_y,
    )
    x_max_y_min_target = box(
        rects_max_x - target_width,
        rects_min_y,
        rects_max_x,
        rects_min_y + target_height,
    )
    target_rects = [
        x_min_y_min_target,
        x_max_y_max_target,
        x_min_y_max_target,
        x_max_y_min_target,
    ]
    target_area_fill = 0
    cutoff_area = 0
    for target in target_rects:
        if intersection(target, union_rects).area > target_area_fill:
            target_area_fill = intersection(target, union_rects).area
            cutoff_area = union_rects.area - target_area_fill
    fill_ratio = target_area_fill / (target_width * target_height)
    overlap_area = get_overlap_area(shapely_rects)
    return fill_ratio, overlap_area, cutoff_area
