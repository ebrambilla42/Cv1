from skimage.transform import AffineTransform, warp
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.io import imread

import math
import matplotlib.pyplot as plt


def get_descriptor(img):
    descriptor_extractor = ORB(n_keypoints=100)

    descriptor_extractor.detect_and_extract(img)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    return keypoints, descriptors


if __name__ == "__main__":
    # load image as grayscale
    im = imread("Elbphilharmonie.jpg", as_gray=True)

    # convert 20 degrees to radians and transform image
    rotation_radians = 20 * (math.pi / 180)
    transform = AffineTransform(
        scale=2.0, rotation=rotation_radians, translation=(-300, -300)
    )
    im2 = warp(im, transform)

    # visualize images
    fig, ax = plt.subplots(2)
    ax[0].imshow(im)
    ax[0].axis("off")
    ax[1].imshow(im2)
    ax[1].axis("off")
    plt.title("Original image (top) vs transformed image (bottom)")
    plt.show()

    # get 100 ORB key points and descriptors
    keypoints_1, descriptors_1 = get_descriptor(im)
    keypoints_2, descriptors_2 = get_descriptor(im2)

    # match descriptors
    matches = match_descriptors(descriptors_1, descriptors_2)

    # visualize matching results
    fig, ax = plt.subplots()
    plot_matches(ax, im, im2, keypoints_1, keypoints_2, matches)
    ax.axis("off")
    ax.set_title("ORB matches of original image (left) vs. transformed image (right)")
    plt.show()

