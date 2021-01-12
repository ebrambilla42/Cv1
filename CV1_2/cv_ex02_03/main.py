"""
Computer Vision I - Assignment 02 
Task 3

Group members:  Emilio Brambilla, Lasse Haffke, Moritz Lahann
"""

"""
Differences to integral image approach:
The integral image approach with center-surround and surround-center
filters difference-of-boxes (DoB) approximates the
difference-of-gaussian (DoG) approach we used here.
Filtering is done on one image resolution vs. multiple
in pyramids. There is effectively only one filter size
used in the integral image approach. Since filters respond
to size of features in the image, this means we may lose out 
on features that our filter size is too large or small for.
Image pyramids essentially emulate different filter sizes by
downsampling the image progressively.

Advantages of using image pyramids:
Image pyramids use multiple image resolutions (progressively more smoothed
and downsampled). This means both large-scale
and small-scale features are taken into account.
They are also computationally less expensive than using
large filters and filter with multiple sizes.
Smoothing and downsampling are cheaper operations than
large convolutions.
"""

# Imports
from skimage.transform.pyramids import pyramid_gaussian
from skimage.transform import resize
from skimage.io import imread, imshow
import numpy as np
import matplotlib.pyplot as plt

# Function for plotting image pyramids
# Shows layers as a single row
# layers are scaled to the same dimensions in the plot
def show_pyramid(pyramid):
    images = []
    titles = []
    for idx, layer in enumerate(pyramid):
        images.append(layer)
        titles.append("Layer {}".format(idx))

    fig, ax = plt.subplots(1, len(images))
    for idx, title in enumerate(titles):
        ax[idx].imshow(images[idx])
        ax[idx].set_title(title)
        ax[idx].set_axis_off()
    return fig


# Function for calculating a contrast pyramid
# Output is an ndarray clipped to [0, 1]
def contrast_pyramid(minuend, subtrahend):
    contrast_pyramid = []
    for idx, m in enumerate(minuend):
        m = np.array(m)
        s = np.array(subtrahend[idx])
        a = m - s
        contrast_pyramid.append(np.clip(a, 0, 1))
    return np.array(contrast_pyramid)


# Function for calculating the feature map of an image pyramid
# Resizes layers to the size of the 1st layer
# Averages layers pixelwise
def feature_map(pyramid):
    feature_map = []
    for idx in range(pyramid.shape[0]):
        if idx > 1:
            r = resize(pyramid[idx], pyramid[1].shape)
            feature_map.append(r)
        else:
            feature_map.append(pyramid[idx])

    # Remove layer 0 (original image)
    feature_map = np.array(feature_map[1:])

    # Average pixelwise
    feature_map = np.mean(feature_map, axis=0)
    return feature_map


# Main function
if __name__ == "__main__":

    # Load image and convert to grayscale
    visual_attention = imread(
        "C:/Users/Moritz Lahann/Desktop/STUDIUM/Module IAS/Computer Vision/cv_exercises/cv_ex02_03/visual_attention.png",
        as_gray=True,
    )
    fig, ax = plt.subplots()
    ax.imshow(visual_attention)
    plt.title("Input image")
    plt.show()

    # Calculate center pyramid
    center_pyramid = tuple(pyramid_gaussian(visual_attention, max_layer=4, sigma=9))

    # Visualize center pyramid
    fig = show_pyramid(center_pyramid)
    fig.suptitle("Center Pyramid")
    plt.show()

    # Calculate surround pyramid
    surround_pyramid = tuple(pyramid_gaussian(visual_attention, max_layer=4, sigma=16))

    # Visualize surround pyramid
    fig = show_pyramid(surround_pyramid)
    fig.suptitle("Surround Pyramid")
    plt.show()

    # Calculate on-off contrast pyramid
    on_off = contrast_pyramid(center_pyramid, surround_pyramid)

    # Calculate off-on contrast pyramid
    off_on = contrast_pyramid(surround_pyramid, center_pyramid)

    # Visualize on-off contrast pyramid
    fig = show_pyramid(on_off)
    fig.suptitle("On-Off Contrast Pyramid")
    plt.show()

    # Visualize off-on contrast pyramid
    fig = show_pyramid(off_on)
    fig.suptitle("Off-On Contrast Pyramid")
    plt.show()

    # Calculate and visualize feature map for on-off
    on_off_map = feature_map(on_off)
    fig, ax = plt.subplots()
    ax.imshow(on_off_map)
    plt.title("On-Off Feature Map")
    plt.show()

    # Calculate and visualize feature map for off-on
    off_on_map = feature_map(off_on)
    fig, ax = plt.subplots()
    ax.imshow(off_on_map)
    plt.title("Off-On Feature Map")
    plt.show()

    # Calculate and show saliency map
    saliency_map = (on_off_map + off_on_map) / 2
    fig, ax = plt.subplots()
    ax.imshow(saliency_map)
    plt.title("Saliency Map")
    plt.savefig("saliency_map.png")
    plt.show()
