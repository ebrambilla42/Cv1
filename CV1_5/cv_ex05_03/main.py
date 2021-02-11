# Group: Emilio Brambilla, Lasse Haffke, Moritz Lahann

from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread
import numpy as np

import matplotlib.pyplot as plt


def slic_visualize(image, n=100, c=10.0):
    superpixels = slic(input_image, n_segments=n, compactness=c)

    plt.imshow(mark_boundaries(input_image, superpixels))
    plt.title(f"SLIC with {n} labels, compactness = {c}")
    plt.show()

    return superpixels


def undersegmentation(ground_truth, superpixels, label):
    # Get the labels in the SLIC image that overlap with the current ground truth label
    overlapping_superpixels = superpixels[ground_truth == label]
    superpixel_labels = np.unique(overlapping_superpixels)

    # Count the number of pixels in the SLIC image that are labeled with an overlapping label from above
    superpixel_area = len(superpixels[np.isin(superpixels, superpixel_labels)])

    # Count the number of pixels in the ground truth with the current label
    ground_truth_area = len(ground_truth[ground_truth == label])

    # Calculate undersegmentation error
    return (superpixel_area - ground_truth_area) / ground_truth_area


if __name__ == "__main__":
    # Read and visualize ground truth and input images
    input_image = imread("cv_ex05_03/0001_rgb.png")
    ground_truth = imread("cv_ex05_03/0001_label.png")

    plt.imshow(input_image)
    plt.title("Input image")
    plt.show()

    plt.imshow(ground_truth)
    plt.title("Ground truth segments")
    plt.show()

    # Run SLIC with skimage default parameters (n=100, c=10.0)
    slic_visualize(input_image)

    # Increase number of superpixels
    # Number of superpixels is increased to capture smaller details in the image
    # (e.g. the tap in the sink)
    slic_visualize(input_image, 250, 10.0)

    # Lowering compactness would lead to severe undersegmentation in some areas
    slic_visualize(input_image, 250, 1.0)

    # While increasing compactness leads to loss of detail as clusters become too square
    # Particularly visible in the top right ceiling area
    slic_visualize(input_image, 250, 50.0)

    # We stick with n=250, c=10 as our best values
    superpixels = slic_visualize(input_image, 250, 10.0)

    errors = []

    # Get all labels in the ground truth image
    for label in np.unique(ground_truth):

        # Filter out zero (unlabeled)
        if label > 0:

            # Calculate undersegmentation error of label
            error = undersegmentation(ground_truth, superpixels, label)
            print(f"The undersegmentation error of label {label} is {error}.")
            errors.append(error)

    # Calculate average error over all labels in ground truth
    print(f"The average undersegmentation error is {np.mean(errors)}.")

    # Increase the number of superpixels
    superpixels = slic_visualize(input_image, 500, 10.0)

    # Calculate average undersegmentation error
    for label in np.unique(ground_truth):
        if label > 0:
            error = undersegmentation(ground_truth, superpixels, label)
            errors.append(error)

    print(
        f"The average undersegmentation error for 500 superpixels is {np.mean(errors)}."
    )

    # Decrease the number of superpixels
    superpixels = slic_visualize(input_image, 50, 10.0)

    # Calculate average undersegmentation error
    for label in np.unique(ground_truth):
        if label > 0:
            error = undersegmentation(ground_truth, superpixels, label)
            errors.append(error)

    print(
        f"The average undersegmentation error for 50 superpixels is {np.mean(errors)}."
    )

    # The average undersegmentation error is lower for higher numbers of superpixels in SLIC (and vice versa).
    # Intuitively, this is because with much smaller superpixels, there is less area that can be outside of the
    # ground truth segment for each superpixel. For example, a superpixel with an area of 20 pixels can produce
    # a much higher undersegmentation error if only slightly overlapping the ground truth segment, compared to a
    # superpixel with an area of 8 pixels.
    # The relative size of superpixels to ground truth segments is especially impactful because of this.

