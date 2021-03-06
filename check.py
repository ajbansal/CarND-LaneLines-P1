# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import math


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform

    Args:
        img (numpy.ndarray): The image to apply canny transform
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def line_slope(line):
    x1, y1, x2, y2 = line[0]
    return float(y2-y1)/(x2-x1)


def line_filter(lines, filter):
    slope_filt_1, tolerance_1 = filter[0]
    slope_filt_2, tolerance_2 = filter[1]
    filtered_lines = [[], []]
    for line in lines:
        slope = line_slope(line)
        if slope_filt_1 - tolerance_1 < slope < slope_filt_1 + tolerance_1:
            filtered_lines[0].append(line)
        elif slope_filt_2 - tolerance_2 < slope < slope_filt_2 + tolerance_2:
            filtered_lines[1].append(line)
    filtered_lines = combine_lines(filtered_lines)
    return filtered_lines


def combine_lines(lines):
    combined_lines = []
    global YMAX
    for line in lines:
        if len(line) == 1:
            combined_lines.append(line)
        else:
            points = []
            for l in line:
                points.append(l[0][:2])
                points.append(l[0][2:])
            (vx, vy, x, y) = cv2.fitLine(np.array(points), cv2.DIST_L12, 0, 0.01, 0.01)
            y1 = np.float32(YMAX/1.6)
            x1 = np.float32(x[0] + ((y1 - y[0]) / vy[0]) * vx[0])

            y2 = np.float32(YMAX)
            x2 = np.float32(x[0] + ((y2 - y[0])/vy[0]) * vx[0])
            combined_lines.append([[x1, y1, x2, y2]])
    return combined_lines


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    lines = line_filter(lines, filter=[[0.5, 0.2], [-0.6, 0.3]])
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=10)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


for path in os.listdir("test_images")[1:]:
    # reading in an image
    image = mpimg.imread(os.path.join("test_images", path))
    # # 1. Convert to gray scale
    # gray = grayscale(image)
    # # 2 Get region of interst
    # interested_region = region_of_interest(gray, np.array([[[60, 539], [487, 195], [930, 539]]], dtype=np.int32))
    # # 3. Only keep white pixels above a threshold
    # white_threshold = 220
    # color_select = np.copy(interested_region)
    # color_select[color_select < white_threshold] = 0
    # color_select[color_select >= white_threshold] = 255
    # # printing out some stats and plotting
    # print('This image is:', type(image), 'with dimesions:', image.shape)
    # plt.imshow(color_select, cmap='gray')
    # plt.show()  # call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    # pass
    global YMAX
    YMAX = np.float32(image.shape[0])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion
    gauss_blur_img = gaussian_blur(gray, 3)
    canny_img = canny(gauss_blur_img, 100, 300)
    interested_region_img = region_of_interest(canny_img, np.array([[[100, 539], [487, 300], [870, 539]]], dtype=np.int32))
    hough_lines_img = hough_lines(interested_region_img, rho=2, theta=np.pi/180, threshold=25, min_line_len=10, max_line_gap=5)
    gray2 = cv2.cvtColor(hough_lines_img, cv2.COLOR_RGB2GRAY) #grayscale conversion
    gray2[gray2 > 0] = 255
    no_zero_indices = hough_lines_img > 0

    no_zero_indices_gradient = np.gradient(gray2, axis=0)
    counter = 0
    vertices_dict = {}
    grads = set()
    for i in no_zero_indices_gradient:
        if any(i):
            vertices_dict[counter] = []
            sub_counter = 0
            for j in i:
                if j:
                    grads.add(j)
                    vertices_dict[counter].append(sub_counter)
                sub_counter += 1
        counter += 1
        pass

    weighted_image = weighted_img(hough_lines_img, image)
    plt.imshow(weighted_image, cmap='gray')
    plt.show()
    pass
