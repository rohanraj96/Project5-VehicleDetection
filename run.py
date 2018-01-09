from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import glob
from queue import Queue
from scipy.ndimage.measurements import label
from sliding_window import *
from features import extract_features
import pickle
import matplotlib.pyplot as plt

class find_cars():

    def __init__(self, ):

        self.cspace = 'YCrCb'
        self.orient = 9
        self.spatial_size = (32, 32)
        self.hist_bins = 32
        self.hist_range = (0, 256)
        self.hog_channel = 'ALL'
        self.pix = 8
        self.cell = 2
        self.path = False
        self.classifier = pickle.load(open('models/linearsvc_ycrcb_all.p', "rb"))
        self.X_scaler = pickle.load(open('models/x_scaler_ycrcb_all.p', "rb"))
        self.count = 0
        self.cumm_heatmap = None
        self.q = Queue()

    def add_heat(self, heatmap, bboxes):

        for bbox in bboxes:

            # print (bbox)
            heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 2

        return heatmap

    def threshold(self, heatmap, threshold):

        heatmap[heatmap <= threshold] = 0

        return heatmap


    def draw_labeled_bboxes(self, img, labels):

        for detection_number in range(1, labels[1]+1):
            nonzero = (labels[0] == detection_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            width = np.max(nonzerox) - np.min(nonzerox)
            height = np.max(nonzeroy) - np.min(nonzeroy)

            # if (width * height > 5000):# & ((width/height > 1) & (width/height < 3)):
            cv2.rectangle(img, bbox[0], bbox[1], (255,0,255), 6)

        return img

    def add_to_heatlist(self, heatmap, frame):

        if(self.count == 0):

            self.cumm_heatmap = np.zeros_like(frame[:,:,0])

        self.cumm_heatmap += heatmap

        if(self.q.qsize() == 10):

            _ = self.q.get()
            self.q.put(heatmap)

        else:

            self.q.put(heatmap)


    def process_image(self, frame):

        framebgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        features = extract_features(frame, self.cspace, self.orient, self.spatial_size, self.hist_bins, self.hist_range, self.pix, self.cell, self.hog_channel, self.path)

        x_start_stop = [700, 1280]
        y_start_stop = [400, 600]
        xy_window = [(64, 64), (96, 96), (128, 128)]
        xy_overlap = (0.5, 0.5)

        total_windows = []
        total_detections = []

        for window_size in xy_window:
        
            window_list = []
            window_list = slide_window(frame, x_start_stop, y_start_stop, window_size, xy_overlap)
            detections = search_windows(frame, window_list, self.classifier, self.X_scaler, color_space = self.cspace, hog_channel = self.hog_channel)
            total_windows.append(window_list)
            # print("Detections: ", detections)
            total_detections.append(detections)
            
        superimposed = np.copy(frame)

        for detection in total_detections:
            if len(detection) > 0:
                superimposed = draw_boxes(superimposed, detection)
                blank = np.zeros_like(superimposed[:,:,0])
                heatmap = self.add_heat(blank, detection)
                # return heatmap
                self.add_to_heatlist(heatmap, frame)
                self.count += 1
        # thresholded_heatmap = self.threshold(heatmap, 1)

        # self.add_to_heatlist(thresholded_heatmap, frame)
        # contour = self.heatmap_history(thresholded_heatmap)

        hotspots = self.threshold(self.cumm_heatmap, 20)
        labels = label(hotspots)

        # labels = label(thresholded_heatmap)
        # if labels[1] > self.detections:
            # self.detections = labels[1]
    
        draw_img = self.draw_labeled_bboxes(np.copy(frame), labels)
        # self.frames += 1

        return draw_img