import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.restoration import denoise_bilateral


def bb_intersection_over_union(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


def get_box_new(box_new):
    x1 = box_new[:, 0]  # x coordinate of the top-left corner
    y1 = box_new[:, 1]  # y coordinate of the top-left corner
    x2 = box_new[:, 2]  # x coordinate of the bottom-right corner
    y2 = box_new[:, 3]  # y coordinate of the bottom-right corner
    return x1, y1, x2, y2


def get_poly(box, text_box):
    poly = np.array(box).astype(np.int32)
    poly_crop = poly.reshape(-1, 1)
    poly_rectangle = poly.reshape(-1, 2)

    poly_text = np.array(text_box[0]).astype(np.int32)
    poly_rectangle_text = poly_text.reshape(-1, 2)
    return poly_crop, poly_rectangle, poly_text, poly_rectangle_text


def image_preprocessing(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 127, 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def get_list_box(detected_data):
    list_box = []
    for item in json.loads(detected_data):
        box = item['box']
        box = np.array(box, dtype=np.float32)
        list_box.append(box)
    list_box = np.array(list_box, dtype=np.float32)
    return list_box


def get_list_text(text_data):
    list_box_text = []
    for item in json.loads(text_data):
        list_box_text.append(item)
    return list_box_text


def get_contours_new(contours, flag_cube):
    contours_new = []
    for contour in contours:
        contour_length = cv2.arcLength(contour, True)
        if 150.0 < contour_length < 350.0:
            metric = (contour_length ** 2) // cv2.contourArea(contour)
            if metric <= 17.0:
                flag_cube = 1
                contours_new.append(contour)
    return contours_new, flag_cube


def get_param_to_size(poly_rectangle, w):
    cm_pixel_cube = 1.27 / w
    mineral_y = poly_rectangle[1][1] - poly_rectangle[0][1]
    mineral_x = poly_rectangle[1][0] - poly_rectangle[0][0]
    mineral_x_in_cm = cm_pixel_cube * mineral_x
    return mineral_x, mineral_y, mineral_x_in_cm, cm_pixel_cube


def hough_theta_ro_to_ab(theta, ro):
    a = -1 / np.tan(theta)
    b = ro / np.sin(theta)
    return a, b


def get_cube_contour(contours, flag_cube, img):

    weight = 0
    contours_new, _ = (contours, 0)
    sorted_contours = sorted(contours_new, key=cv2.contourArea, reverse=True)

    for (i, c) in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(c)
        cropped_contour = img[y:y + h, x:x + w]
        if 50 < cropped_contour.shape[0] < 100 and 50 < cropped_contour.shape[1] < 100:
            plt.imshow(cropped_contour)
            plt.show()
            contours_crop = image_preprocessing(cropped_contour)

            if len(contours_crop) <= 4:
                pairs_cnt = hough_viz(cropped_contour)
                if pairs_cnt == 2:
                    flag_cube = 1
                    weight = w
    return weight, flag_cube


def hough_viz(img):
    img = rgb2gray(img)
    img = denoise_bilateral(img, win_size=1)
    img = canny(img, sigma=1)

    fig, axes = plt.subplots(1, 4, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(img)

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(img, theta=tested_angles)

    ax[1].imshow(img, cmap=cm.gray)
    ax[1].set_title('Input image')
    ax[1].set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]

    ax[2].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[2].set_title('Hough transform')
    ax[2].set_xlabel('Angles (degrees)')
    ax[2].set_ylabel('Distance (pixels)')
    ax[2].axis('image')

    ax[3].imshow(img, cmap=cm.gray)
    ax[3].set_ylim((img.shape[0], 0))
    ax[3].set_axis_off()
    ax[3].set_title('Detected lines')

    peaks = hough_line_peaks(h, theta, d)

    angles_list = list()
    for _, angle, dist in zip(*peaks):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[3].axline((x0, y0), slope=np.tan(angle + np.pi / 2))
        degree = np.cos(angle) * 180
        angles_list.append(degree)

    angles_list = np.sort(np.array(angles_list).astype(int))

    pairs_cnt = 0
    prev_is_pair = False
    max_degree = 7
    for i in range(1, len(angles_list)):
        diff = angles_list[i] - angles_list[i - 1]

        if not prev_is_pair:
            if diff < max_degree:
                pairs_cnt += 1

        if diff > max_degree:
            prev_is_pair = False
        else:
            prev_is_pair = True

    if pairs_cnt == 2 and len(angles_list) <= 15:
        plt.tight_layout()
        plt.show()
        return pairs_cnt
    else:
        return 1
