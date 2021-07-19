#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 12:46:49 2021

@author: ruizvilj
"""

import cv2 as cv
import numpy as np
import math

class CannyFilter():

    def __init__(self, img_name):
        self.img_name = img_name
        self.original_img = cv.imread(self.img_name)

        if self.original_img is None:
            raise Exception("[ERROR] Image not found in path.")

    def get_original_image(self):
        return self.original_img

    # Return the X and Y gradients of the image
    def sobelAlgorithm(self, input_image):
        img_height, img_width = self.original_img.shape

        # horizontal kernel mask
        kernel_x = \
            [
                [ -1, 0, 1],
                [ -2, 0, 2],
                [ -1, 0, 1]
            ]

        # Vertical kernel mask
        kernel_y =\
            [
                [-1,-2,-1],
                [ 0, 0, 0],
                [ 1, 2, 1]
            ]


        # Kernel dimensions
        kernel_size = 3
        kernel_radio = kernel_size // 2

        # Used to store (x,y) gradients
        gradient_x = np.zeros_like(self.original_img, np.float64)
        gradient_y = np.zeros_like(self.original_img, np.float64)

        for x in range(img_height):
            for y in range(img_width):
                horizon_pixel_sum, vertical_pixel_sum = 0, 0
                full_kernel_iteration = True
                for i in range(-kernel_radio, kernel_radio+1):
                    for j in range(-kernel_radio, kernel_radio+1):
                        # Target coordinate for original image
                        tg_x, tg_y = x + i, y + j

                        # Target coordinates for kernel
                        ktg_x, ktg_y = i + kernel_radio, j + kernel_radio

                        if tg_x < 0 or tg_x >= img_height or tg_y < 0 or tg_y >= img_width:
                            full_kernel_iteration = False
                            continue

                        # Calculate gradient X
                        horizon_pixel_sum += kernel_x[ktg_x][ktg_y] * input_image[tg_x, tg_y]

                        # Calculate gradient Y
                        vertical_pixel_sum += kernel_y[ktg_x][ktg_y] * input_image[tg_x, tg_y]

                if full_kernel_iteration:
                    gradient_x[x, y] = float(horizon_pixel_sum)
                    gradient_y[x, y] = float(vertical_pixel_sum)

                # Formula to merge results = sqrt(img1_pixel ^ 2 + img2_pixel ^ 2), btw this is the magnitude
                #new_pixel = int(math.sqrt(pow(horizon_pixel_sum, 2) + pow(vertical_pixel_sum, 2)))
                #output_img[x, y] = new_pixel

        return gradient_x, gradient_y

    def nonMaximumSuppresion(self, magnitudes, angles):
        img_height, img_width = self.original_img.shape

        for x in range(img_height):
            for y in range(img_width):
                gradient_angle = angles[x, y]
                gradient_angle = abs(gradient_angle - 180) if abs(gradient_angle) > 180 else abs(gradient_angle)

                # Now select the neighbours based on gradient direction
                neighb_x_1, neighb_y_1, neighb_x_2, neighb_y_2 = 0, 0, 0, 0

                # Neighbours in the X axis
                if gradient_angle <= 22.5:
                    neighb_x_1, neighb_y_1 = x, y - 1
                    neighb_x_2, neighb_y_2 = x, y + 1
                # Neighbours in the right diagonal (\) from picture perspective
                elif gradient_angle > 22.5 and gradient_angle <= (22.5 + 45.0):
                    neighb_x_1, neighb_y_1 = x - 1, y - 1
                    neighb_x_2, neighb_y_2 = x + 1, y + 1
                # Neighbours in the Y axis
                elif gradient_angle > (22.5 + 45.0) and gradient_angle <= (22.5 + 90.0):
                    neighb_x_1, neighb_y_1 = x - 1, y
                    neighb_x_2, neighb_y_2 = x + 1, y
                # Neighbours in the left diagonal (/) from picture perspective
                elif gradient_angle > (22.5 + 90) and gradient_angle <= (22.5 + 135.0):
                    neighb_x_1, neighb_y_1 = x + 1, y - 1
                    neighb_x_2, neighb_y_2 = x - 1, y + 1
                # Again comple cycle, that means X axis
                elif gradient_angle > (22.5 + 135) and gradient_angle <= (22.5 + 180.0):
                    neighb_x_1, neighb_y_1 = x, y - 1
                    neighb_x_2, neighb_y_2 = x, y + 1

                if img_height > neighb_x_1 > 0 and img_width > neighb_y_1 > 0:
                    if magnitudes[x, y] < magnitudes[neighb_x_1, neighb_y_1]:
                        magnitudes[x, y] = 0
                        continue

                if img_height > neighb_x_2 > 0 and img_width > neighb_y_2 > 0:
                    if magnitudes[x, y] < magnitudes[neighb_x_2, neighb_y_2]:
                        magnitudes[x, y] = 0

        return magnitudes


    def applyFilter(self):
        # Convert to gray scale to easy manipulation
        self.original_img = cv.cvtColor(self.original_img, cv.COLOR_BGR2GRAY)

        # Let's remove some noisy using a Gausassian filter
        output_img = cv.GaussianBlur(self.original_img, (5, 5), 1.4)

        # Let's get the gradients provided by Sobel algorithm
        gx, gy = self.sobelAlgorithm(output_img)

        # Convert cartesian coordinates to polar
        magnitudes, angles = cv.cartToPolar(gx, gy, angleInDegrees=True)

        magnitudes = self.nonMaximumSuppresion(magnitudes, angles)

        # Minimun and maximun thresholds
        threshold_min = np.max(magnitudes) * 0.1
        threshold_max = np.max(magnitudes) * 0.5

        img_height, img_width  = self.original_img.shape
        for x in range(img_height):
            for y in range(img_width):
                grad_mag = magnitudes[x, y]
                if grad_mag < threshold_min:
                    magnitudes[x, y] = 0
                else:
                    magnitudes[x, y] = 255

        return magnitudes

# Apply the closing morphological filter to the input image
# and return an image with a Grayscale
def closing_filter(input_image):
    morph_kernel = cv.getStructuringElement(cv.MORPH_OPEN, (5, 5))
    image_with_closing_filter = cv.morphologyEx(input_image, cv.MORPH_CLOSE, morph_kernel)

    # Important return the image in Grayscale (8 bits)
    return image_with_closing_filter.astype(np.uint8)

# Count the figures found in the image and return the input image with the labels
def countFigures(input_image, image_with_edges):

    triangles_count, rectables_count, pentagon_count = 0, 0, 0
    hexagon_count, octagon_count, circle_count = 0, 0, 0

    # Find the image contours
    image_contours, _ = cv.findContours(image_with_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for count in image_contours:
        # Epsilon is the approximation accuracy
        epsilon = 0.01 * cv.arcLength(count, True)
        approximations = cv.approxPolyDP(count, epsilon, True)

        cv.drawContours(input_image, [approximations], 0, (0), 2)

        # Name of the detected shapes are written on the image in spanish :)
        i, j = approximations[0][0]
        if len(approximations) == 3:
            cv.putText(input_image, "Triangle", (i, j), cv.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)
            triangles_count += 1
        elif len(approximations) == 4:
            cv.putText(input_image, "Rectangle", (i, j), cv.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)
            rectables_count += 1
        elif len(approximations) == 5:
            cv.putText(input_image, "Pentagon", (i, j), cv.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)
            pentagon_count += 1
        elif len(approximations) == 6:
            cv.putText(input_image, "Hexagon", (i, j), cv.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)
            hexagon_count += 1
        elif len(approximations) == 8:
            cv.putText(input_image, "Octagon", (i, j), cv.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)
            octagon_count += 1
        # Let's consider anything above 12 verteces as circle
        elif len(approximations) >= 12:
            cv.putText(input_image, "Circle", (i, j), cv.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)
            circle_count += 1


    print("Triangles:", triangles_count)
    print("Rectangles: ", rectables_count)
    print("Pentagons: ", pentagon_count)
    print("hexagons: ", hexagon_count)
    print("Octagons: ", octagon_count)
    print("Circles: ", circle_count)

    # Return the input image with the labels on it
    return input_image


print("CAPTCHA SOLVER")
images_arr = ["figures4.jpg", "figures5.png", "figures2.png"]
test_case = 1
for image_name in images_arr:

    print("Case #" + str(test_case),", Image: ", image_name, sep='')

    # Edge detection by Canny algorithm
    cannyFilter = CannyFilter(image_name)
    image_with_edges = cannyFilter.applyFilter()
    original_image = cv.imread(image_name)

    # Apply closure filter
    image_with_edges = closing_filter(image_with_edges)

    # Count the figures
    image_with_labels = countFigures(original_image, image_with_edges)

    cv.imshow("Original image", cannyFilter.get_original_image())
    cv.imshow("Image with edges", image_with_edges)
    cv.imshow("Image with labels", image_with_labels)

    print("Press any key to continue\n")
    cv.waitKey(60000)
    cv.destroyAllWindows()
    test_case += 1

