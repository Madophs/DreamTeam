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
        self.original_img = None

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
                    neighb_x_2, neighb_y_2 = x, y - 1
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
        # Let's apply a grayscaling for easy manipulation
        self.original_img = cv.imread(self.img_name)

        if self.original_img is None:
            raise Exception("[ERROR] Image not found in path.")

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
        return magnitudes


images_arr = ["house.jpg", "lambo.png", "transit1.jpg"]
for image_name in images_arr:
    cannyFilter = CannyFilter(image_name)
    output_img = cannyFilter.applyFilter()

    cv.imshow("Original image", cannyFilter.get_original_image())
    cv.imshow("Output image", output_img)

    print("Press any key to continue")
    cv.waitKey(60000)
    cv.destroyAllWindows()

