import math
import sys
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO if RGB repeat 3 times for each color
    # output = (img_height - kernel_height + 1) * (img_width - kernel_width + 1)
    # want output height = img_height, so need to pad img with kernel_height - 1 (and same for width)
    # therefore for output_height to be = to img_height, new height of img = img_height + kernel_height - 1, need to pad with (kernel_height - 1)
    # this works since m and n (kernel dimensions) are odd
    new_img = np.zeros_like(img)
    kernel_height = np.shape(kernel)[0]
    kernel_width = np.shape(kernel)[1]
    img_height = np.shape(img)[0]
    img_width = np.shape(img)[1]
    h_padding = (kernel_height - 1) // 2 # need padding on both sides
    w_padding = (kernel_width - 1) // 2
    if len(img.shape) == 3:
        for channel in range(0, 3):
            padded_img = np.pad(img[:, :, channel], ((h_padding, h_padding), (w_padding, w_padding)), mode='constant')
            for i in range(img_height):
                for j in range(img_width):
                    new_img[i, j, channel] = np.sum(kernel * (padded_img[i:i + kernel_height, j:j+kernel_width]))
    else: # grayscale
        padded_img = np.pad(img, ((h_padding, h_padding), (w_padding, w_padding)), mode='constant')
        for i in range(img_height):
            for j in range(img_width):
                new_img[i, j] = np.sum(kernel * (padded_img[i:i + kernel_height, j:j+kernel_width]))
    return new_img
    raise Exception("TODO in hybrid.py not implemented")


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    return cross_correlation_2d(img, kernel)
    raise Exception("TODO in hybrid.py not implemented")

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    kernel = np.zeros((height, width))
    const = 1 / (2 * math.pi * (sigma**2))
    denom = 2 * (sigma**2)
    sum = 0.0
    for i in range(height):
        for j in range(width):
            x = j - (width // 2)
            y = i - (height // 2)
            kernel[i, j] = const * math.exp(-((x**2) + (y**2))/ denom)
            sum += kernel[i, j]
    # print(sum)
    kernel /= sum # normalize
    return kernel
    raise Exception("TODO in hybrid.py not implemented")

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''    

    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel)
    raise Exception("TODO in hybrid.py not implemented")

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return img - low_pass(img, sigma, size)
    raise Exception("TODO in hybrid.py not implemented")

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

