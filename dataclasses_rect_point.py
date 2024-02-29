from dataclasses import dataclass
from typing import Optional


@dataclass
class Point:
    """A point in 2D space."""

    x: int
    y: int


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
