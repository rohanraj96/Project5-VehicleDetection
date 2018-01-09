import numpy as np
import cv2
from features import extract_features

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            window_list.append(((startx, starty), (endx, endy)))

    return window_list

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):

    # imcopy = np.copy(img)
    # bboxes = np.array(bboxes, dtype = np.uint8)
    for bbox in bboxes:
        # print (bbox)
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)

    return img

def search_windows(image, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0):

    on_windows = []

    # image_shrunk = cv2.resize(image, (int(image.shape[1]//1.5), int(image.shape[0]//1.5)))
    # image_smallest = cv2.resize(image, (int(image.shape[1]//3), int(image.shape[0]//3)))
    # scaled = [image, image_shrunk, image_smallest]
    # count = 0
    # for img in scaled:

    for window in windows:
            # print(window)
        if (window[1][0] <= image.shape[1]) & (window[1][1] <= image.shape[0]): 
            test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            features = extract_features(test_img, cspace=color_space, orient=orient,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    hist_range=hist_range, pix=pix_per_cell, 
                                    cell=cell_per_block, hog_channel=hog_channel, path=False)
            # print(np.array(features))
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            prediction = clf.predict(test_features)
            if (prediction == 1):# & (count == 0):
                # print(window)
                on_windows.append(window)

        #         elif (prediction == 1) & (count == 1):
        #             top_left = window[0]
        #             bottom_right = window[1]
        #             x1 = top_left[0]
        #             y1 = top_left[1]
        #             x2 = bottom_right[0]
        #             y2 = bottom_right[1]
        #             _lambda = np.sqrt(1.5)
        #             del_l = img.shape[1]*(_lambda - 1)
        #             del_b = img.shape[0]*(_lambda - 1)
        #             # print(top_left)
        #             # print(bottom_right)
        #             # print(x1)
        #             x1_new = int(x1 - del_l/2)
        #             x2_new = int(x2 + del_l/2)
        #             y1_new = int(y1 - del_b/2)
        #             y2_new = int(y2 + del_b/2)
        #             window = ((x1_new, y1_new), (x2_new, y2_new))
        #             on_windows.append(window)

        #         elif (prediction == 1) & (count == 2):
        #             top_left = window[0]
        #             bottom_right = window[1]
        #             x1 = top_left[0]
        #             y1 = top_left[1]
        #             x2 = bottom_right[0]
        #             y2 = bottom_right[1]
        #             _lambda = np.sqrt(3)
        #             del_l = img.shape[1]*(_lambda - 1)
        #             del_b = img.shape[0]*(_lambda - 1)
        #             # print(top_left)
        #             # print(bottom_right)
        #             # print(x1)
        #             x1_new = int(x1 - del_l/2)
        #             x2_new = int(x2 + del_l/2)
        #             y1_new = int(y1 - del_b/2)
        #             y2_new = int(y2 + del_b/2)
        #             window = ((x1_new, y1_new), (x2_new, y2_new))
        #             on_windows.append(window)

        # # print('\n\n\n\n\n')
        # # print("###    COUNT = %d   ####"%(count))
        # # print('\n\n\n\n\n')
        # count += 1
            
    # print(on_windows)
    return on_windows

