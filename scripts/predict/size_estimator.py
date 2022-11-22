import numpy as np
import pandas as pd
import cv2
import csv

import utils_size


THRESHOLD_IOU = 0.10

list_path_final = []
list_path_crop = []

df_text_boxes = pd.read_csv('../pavlov/data_text_boxes.csv')  # DEBUG
df = pd.read_csv('../pavlov/data_minerals_boxes.csv')  # DEBUG

with open('size_centimeter_minerals.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['image_path',
                     'contour_analysis_mineral_x',
                     'contour_analysis_mineral_y',
                     'contour_analysis_table_error',
                     'contour_analysis_cm_pixel_cube',
                     'contour_analysis_mineral_x_in_cm',
                     'variety',
                     'satellites'])

    for detected_data, image_path, index, text_data, variety, satellites in zip(df['detected_data'], df['path_final'],
                                                                                range(df['detected_data'].shape[0]),
                                                                                df_text_boxes['text_boxes'],
                                                                                df_text_boxes['variety'],
                                                                                df_text_boxes['satellites']):

        image_path = image_path.replace('/home/jovyan/minerals/', '/home/jovyan/minerals_resize/')  # DEBUG

        flag_cube = 0
        flag_mineral = 0

        try:

            img = cv2.imread(image_path)
            list_box = utils_size.get_list_box(detected_data)
            list_box_text = utils_size.get_list_text(text_data)

            if len(list_box) > 0:
                x1, y1, x2, y2 = utils_size.get_box_new(list_box)

                if len(list_box_text) > 0:
                    for box in list_box:
                        for text_box in list_box_text:
                            text_box = np.array(text_box, dtype=np.int32)
                            text_box = text_box.reshape(-1, 4)
                            iou = utils_size.bb_intersection_over_union(box, text_box[0])
                            poly = np.array(box).astype(np.int32)
                            poly_crop = poly.reshape(-1, 1)
                            poly_rectangle = poly.reshape(-1, 2)

                            if iou < THRESHOLD_IOU:

                                poly_crop, poly_rectangle, poly_text, poly_rectangle_text = \
                                    utils_size.get_poly(box, text_box)
                                flag_mineral = 1
                                contours = utils_size.image_preprocessing(img)
                                utils_size.hough_viz(img)
                                contours_new, flag_cube = utils_size.get_contours_new(contours, flag_cube)
                                w, flag_cube = utils_size.get_cube_contour(contours, flag_cube, img)
                            else:
                                _, poly_rectangle, poly_text, poly_rectangle_text = utils_size.get_poly(box, text_box)

                        img_k = img[int(min(y1)) - 10:int(max(y2) + 10), int(min(x1)) - 10:int(max(x2)) + 10]

                    if flag_cube == 1 and flag_mineral == 1 and w > 0:
                        list_path_final.append(image_path)
                        mineral_x, mineral_y, mineral_x_in_cm, cm_pixel_cube = utils_size.get_param_to_size(
                            poly_rectangle, w)
                        writer.writerow([image_path, int(mineral_x), int(mineral_y), int(0), float(cm_pixel_cube),
                                         int(mineral_x_in_cm), variety, satellites])
                else:
                    img_n = img[int(min(y1)) - 10:int(max(y2) + 10), int(min(x1)) - 10:int(max(x2)) + 10]

        except:
            print('ERROR', image_path)
