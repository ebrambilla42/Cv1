#!/usr/bin/env python
from sys import argv
from skimage import io, feature, transform
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

mark_diameter_mm = 24.5
mm_in_px_res = 0.12
radius_px = (mark_diameter_mm / 2) / mm_in_px_res


def main():
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    (ax1, ax2, ax3) = axes
    for ax in axes:
        ax.axis("off")

    input_path = argv[1]
    grey_image = io.imread(input_path, as_gray=True)
    edges = feature.canny(grey_image)
    hough_circles = transform.hough_circle(edges, radius_px)
    accums, cx, cy, _ = transform.hough_circle_peaks(
        hough_circles, [radius_px for x in hough_circles], total_num_peaks=2
    )

    for center in zip(cx, cy):
        circle = Circle(center, radius_px, fill=False, color="red")
        ax3.add_artist(circle)

    ax1.set_title("Canny-Edge")
    ax2.set_title("Hough-Transform")
    ax3.set_title("Detected circles")
    ax1.imshow(edges)
    ax2.imshow(hough_circles[0])
    ax3.imshow(grey_image, cmap="gray")
    plt.savefig("visualization.png")


if __name__ == "__main__":
    main()
