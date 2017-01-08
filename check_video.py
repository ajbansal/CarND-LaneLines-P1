# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform

    Args:
        high_threshold (int): High threshold of canny transform
        low_threshold (int): Low threshold value of canny transform
        img (numpy.ndarray): Image object for canny tansform

    Returns:
        numpy.ndarray: And image with canny transform applied
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel

    Args:
        kernel_size (int): Kernel size for gaussian blur
        img (numpy.ndarray): Image object

    Returns:
        numpy.ndarray: image with gaussian blur applied
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    Args:
        img (numpy.ndarray): Image on which to apply the logic
        vertices (numpy.ndarray): The three vertices of the polygon in an nd array of
                                  shape (1, n, 2) n being number of vertices

    Returns:
        numpy.ndarray: The image
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
    """
    To give slope of a line defined as two points on the line

    Args:
        line (numpy.ndarray): The two vertices of the line

    Returns:
        numpy.float32: Returns the slope of the line
    """
    x1, y1, x2, y2 = line[0]
    return np.float32((y2-y1)/(x2-x1))


def line_filter(lines, filter):
    """
    The function to only keep lines that meet the slope criteria.
    Helps filter out unwanted lines that do not meet the slope params

    Args:
        filter (List[List[float]]): Filter are the two filter params for right and left lane lines
                                    [[slope1, tol1],[slope2, tol2]]
        lines (numpy.ndarray): All the lines returned by the hough transform
                               In the shape of (n,1,4) n-> total lines, and 4 -> x1,y1,x2,y2

    Returns:
        numpy.ndarray : Filtered lines with just two lines depicting right and left lane
    """

    # Unpack filter params
    slope_filt_1, tolerance_1 = filter[0]
    slope_filt_2, tolerance_2 = filter[1]
    filtered_lines = [[], []]

    # Loop through all lines
    for line in lines:
        slope = line_slope(line)
        # Group lines based on the which slope param they fit in
        if slope_filt_1 - tolerance_1 < slope < slope_filt_1 + tolerance_1:
            filtered_lines[0].append(line)
        elif slope_filt_2 - tolerance_2 < slope < slope_filt_2 + tolerance_2:
            filtered_lines[1].append(line)

    # Combine all the lines to get just two lines, one for left and another for right.
    filtered_lines = combine_lines(filtered_lines)

    # Return combined filtered lines
    return filtered_lines


def combine_lines(lines):
    """
    Combines all right lanes into one right lane and all left lanes to one left lane
    Uses cv2.fitline trying to minimize distance between all lines

    Args:
        lines (List[List[numpy.ndarray]]): List of lines in two lists seperated as right and left

    Returns:
        List[List[List[numpy.float32]]]: Returns right and left lane line
    """
    combined_lines = []

    # Global ymax for the shape of image
    global YMAX, YMIN

    # Loop through all lines
    for line in lines:

        # If only one line then nothing to combine
        if len(line) == 1:
            combined_lines.append(line)
        else:
            points = []
            for l in line:
                points.append(l[0][:2])
                points.append(l[0][2:])

            # Use cv2.fitline to combine all points
            (vx, vy, x, y) = cv2.fitLine(np.array(points), cv2.DIST_L12, 0, 0.01, 0.01)

            # First point is Y cordinate with YMIN as we done need lanes drawn on the entire image
            y1 = YMIN
            x1 = np.float32(x[0] + ((y1 - y[0]) / vy[0]) * vx[0])

            # Second point is with Y coordinate as Y max as the lane starts from the bottom of the image
            y2 = np.float32(YMAX)
            x2 = np.float32(x[0] + ((y2 - y[0])/vy[0]) * vx[0])

            # Add the combined lane info
            combined_lines.append([[x1, y1, x2, y2]])

    # return the combined lines
    return combined_lines


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.

    Args:
        rho (int): The rho param for hough lines
        theta (float): The theta param for hough lines
        threshold (int): The threshold param for hough lines
        min_line_len (int): The min line gap param for hough lines
        max_line_gap (int): The max line gap param for hough lines
        img (numpy.ndarray): The image for hough lines

    Returns:
        numpy.ndarray: The image with hough lines drawn
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


def process_image(image):
    """
    To process each image of the frame and add lane lines

    Args:
        image (numpy.ndarray): The image on which to draw lane lines

    Returns:
        numpy.ndarray: Image with lane lines drawn
    """

    global YMAX, YMIN

    # Params for how long the lane line should be
    YMAX = np.float32(image.shape[0])
    YMIN = np.float32(YMAX/1.6)

    # Convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Add gaussian blur
    gauss_blur_img = gaussian_blur(gray, 3)

    # Add canny transform
    canny_img = canny(gauss_blur_img, 100, 300)

    # Select only the region we want, in this case a triangle in the bottom in which we know the lane lines exist
    interested_region_img = region_of_interest(canny_img, np.array([[[100, 539], [487, 300], [870, 539]]], dtype=np.int32))

    # Draw hough lines and extrapolate them to entire lane
    hough_lines_img = hough_lines(interested_region_img, rho=2, theta=np.pi/180, threshold=35, min_line_len=10, max_line_gap=5)

    # Add lines to original image
    weighted_image = weighted_img(hough_lines_img, image)

    # Return image with lines drawn
    return weighted_image

if __name__ == "__main__":

    # All input videos
    video1 = "solidWhiteRight.mp4"
    video2 = "solidYellowLeft.mp4"

    # Name of new video
    for video in [video1, video2]:
        video_new = "{}_new.mp4".format(video[:video.rfind(".")])

        print("Loading {}".format(video))

        # Load the video
        clip1 = VideoFileClip(video)

        # Give the function to process all images
        new_video_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!

        print("Processing {}".format(video_new))

        # Output the new file
        new_video_clip.write_videofile(video_new, audio=False)
