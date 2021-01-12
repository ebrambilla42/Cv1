"""
Computer Vision I - Assignment 02 
Task 1

Group members:  Emilio Brambilla, Lasse Haffke, Moritz Lahann
"""

from skimage import io
from skimage import feature
import numpy as np
from skimage import color
import matplotlib.pyplot as plt

expected_template_position = (118, 82)


def main():
    img_test = color.rgb2gray(io.imread("coco264316.jpg"))
    img_template = color.rgb2gray(io.imread("coco264316clock.jpg"))
    correlation_image = feature.match_template(img_test, img_template)
    ij = np.unravel_index(np.argmax(correlation_image), correlation_image.shape)
    x, y = ij[::-1]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(correlation_image)
    ax.set_axis_off()
    ax.set_title("`match_template`\ncorrelation_image")
    # highlight matched region
    ax.autoscale(False)
    ax.plot(x, y, "o", markeredgecolor="r", markerfacecolor="none", markersize=10)

    print(f"correct position found: {expected_template_position == (x,y)}")
    plt.show()
    plt.savefig("correlation_image.png")

    img_template = np.fliplr(img_template)
    io.imsave("flipped_template.png", img_template)
    correlation_image = feature.match_template(img_test, img_template)
    ij = np.unravel_index(np.argmax(correlation_image), correlation_image.shape)
    x, y = ij[::-1]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(correlation_image)
    ax.set_axis_off()
    ax.set_title("`match_template_flipped`\ncorrelation_image")
    # highlight matched region
    ax.autoscale(False)
    ax.plot(x, y, "o", markeredgecolor="r", markerfacecolor="none", markersize=10)

    print(
        f"correct position found for flipped template: {expected_template_position == (x,y)}"
    )
    plt.show()
    plt.savefig("correlation_image_flipped.png")


if __name__ == "__main__":
    main()
