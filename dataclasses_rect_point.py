from dataclasses import dataclass
from typing import Optional


@dataclass
class Point:
    '''A point in 2D space.'''
    x: int
    y: int
    
@dataclass
class Rectangle:
    '''A rectangle defined by its lower left corner, width, height, and rotation.'''
    width: int
    height: int
    rotation: int = 0
    lower_left: Optional[Point] = None
