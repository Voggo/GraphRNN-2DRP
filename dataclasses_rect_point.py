from dataclasses import dataclass
from typing import Optional


@dataclass
class Point:
    """A point in 2D space."""

    x: int
    y: int

    def __add__(self, other: "Point") -> "Point":
        if not isinstance(other, Point):
            raise ValueError("You can only add another Point to this Point")
        return Point(self.x + other.x, self.y + other.y)

    def __sub__ (self, other: "Point") -> "Point":
        if not isinstance(other, Point):
            raise ValueError("You can only subtract another Point from this Point")
        return Point(self.x - other.x, self.y - other.y)

    def __round__(self, n = None) -> "Point":
        return Point(round(self.x, n), round(self.y, n))
@dataclass
class Rectangle:
    """A rectangle defined by its lower left corner, width, height, and rotation."""

    width: int
    height: int
    rotation: int = 0
    lower_left: Optional[Point] = None

    def __hash__(self) -> int:
        if self.lower_left is None:
            return hash((self.width, self.height, self.rotation))
        return hash(
            (
                self.width,
                self.height,
                self.rotation,
                self.lower_left.x,
                self.lower_left.y,
            )
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Rectangle):
            return False
        if self.lower_left is None:
            return (
                self.width == __value.width
                and self.height == __value.height
                and self.rotation == __value.rotation
            )
        return (
            self.width == __value.width
            and self.height == __value.height
            and self.rotation == __value.rotation
            and self.lower_left == __value.lower_left
        )
