import pytest

from dataclasses_rect_point import Rectangle, Point
from plot_rects import plot_rects
from generator import get_line_breaks, generate_rects

class TestGetLineBreaks:

    # Returns a list of n_breaks random line breaks within the given line range.
    def test_returns_list_of_random_line_breaks(self):
        line = 10
        n_breaks = 5
        result = get_line_breaks(line, n_breaks)
        print(result)
        assert isinstance(result, list)
        assert len(result) == n_breaks
        assert all(0 <= break_point <= line for break_point in result)

    # Returns a list of 0 line breaks when n_breaks is 0.
    def test_returns_empty_list_when_n_breaks_is_0(self):
        line = 10
        n_breaks = 0
        result = get_line_breaks(line, n_breaks)
        print(result)
        assert isinstance(result, list)
        assert len(result) == 0

    # Returns a list of 1 line break when n_breaks is 1.
    def test_returns_list_of_1_line_break_when_n_breaks_is_1(self):
        line = 10
        n_breaks = 1
        result = get_line_breaks(line, n_breaks)
        print(result)
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(0 <= break_point <= line for break_point in result)

    # Returns an empty list when line is 0.
    def test_returns_empty_list_when_line_is_0(self):
        line = 0
        n_breaks = 5
        result = get_line_breaks(line, n_breaks)
        print(result)
        assert isinstance(result, list)
        assert len(result) == 0

    # Returns a list of line breaks with values ranging from 0 to line (inclusive).
    def test_returns_list_of_line_breaks_within_range(self):
        line = 10
        n_breaks = 5
        result = get_line_breaks(line, n_breaks)
        print(result)
        assert isinstance(result, list)
        assert len(result) == n_breaks
        assert all(0 <= break_point <= line for break_point in result)


# class TestGenerateRects:

#     def test_generate_rects_expected_number_of_rectangles(self):
#         # Arrange
#         width = 10
#         height = 10
#         n_breaks = 2

#         # Act
#         result = generate_rects(width, height, n_breaks)
        
#         # Plot the rectangles
#         plot_rects(
#             result,
#             ax_lim=width,
#             ay_lim=height,
#             filename="test_generate_rects_expected_number_of_rectangles.png",
#             show=False,
#         )

#         # Assert
#         assert len(result) == (n_breaks + 1) ** 2
