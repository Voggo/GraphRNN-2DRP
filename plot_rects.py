import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as plt_Rectangle
from typing import List
from dataclasses_rect_point import Rectangle, Point


def get_plt_rects(rects: List[Rectangle]):
    """Return a list of matplotlib Rectangle patches from a list of Rectangle dataclasses."""
    plt_rects = []
    for rect in rects:
        try:
            if rect.lower_left is None:
                raise ValueError(
                    f"Rectangle ({rect}) did not have a lower_left attribute."
                )
            plt_rect = plt_Rectangle(
                (rect.lower_left.x, rect.lower_left.y) if rect.lower_left else (0, 0),
                rect.width,
                rect.height,
                fill=True,
                color="blue",
                ec="black",
                alpha=0.7,
            )
            plt_rects.append(plt_rect)
        except ValueError as e:
            e = str(e)
    return plt_rects


def plot_rects(
    rects: List[Rectangle],
    ax_lim=5,
    ay_lim=5,
    ax_min=0,
    ay_min=0,
    filename="rects.png",
    show=True,
):
    """Plot a list of Rectangle dataclasses."""
    _, ax = plt.subplots()
    ax.set_xlim(ax_min, ax_lim)
    ax.set_ylim(ay_min, ay_lim)
    plt_rects = get_plt_rects(rects)
    for plt_rect in plt_rects:
        ax.text(
            plt_rect.get_x() + plt_rect.get_width() / 2,
            plt_rect.get_y() + plt_rect.get_height() / 2,
            f"{plt_rects.index(plt_rect)}",
            ha="center",
            va="center",
            color="black",
        )
        ax.add_patch(plt_rect)
    plt.savefig(
        f"plots_img/{filename}",
    )  # Save the figure before showing it
    if show:
        plt.show()


if __name__ == "__main__":
    test_rects = [
        Rectangle(1, 1, lower_left=None),
        Rectangle(1, 1, lower_left=Point(0, 1)),
        Rectangle(1, 1, lower_left=Point(1, 0)),
        Rectangle(1, 1, lower_left=Point(1, 1)),
        Rectangle(1, 1, lower_left=Point(2, 2)),
    ]
    plot_rects(test_rects, ax_lim=3, ay_lim=3)
