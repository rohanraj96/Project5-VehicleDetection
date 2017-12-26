from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import glob
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
        self.last_frames = []
        self.history = []
        self.detections = 0
        self.frames = 0


    def add_heat(self, heatmap, bboxes):

        for bbox in bboxes:
            heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

        return heatmap

    def threshold(self, heatmap, threshold = 2):
        
        heatmap[heatmap <= threshold] = 0

        return heatmap


    def buffer(self, this_frame, detection_number):

        if self.frames == 22:
            self.frames = 0
            self.last_frames = []
            self.history = []

      
        if len(self.last_frames) >= detection_number: #Check if we have any previous frames of this label
            self.last_frames[detection_number - 1].append(this_frame)
            self.history[detection_number - 1] = self.last_frames[detection_number - 1][-1]

        else:
            self.last_frames.append([this_frame])
            self.history.append([this_frame])


    def draw_labeled_bboxes(self, img, labels):

        for detection_number in range(1, labels[1]+1):
            nonzero = (labels[0] == detection_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            self.buffer(bbox, detection_number)
            smoothed = np.array(self.last_frames[detection_number - 1]).mean(axis = 0)
            left = tuple(map(int, tuple(smoothed[0])))
            right = tuple(map(int, tuple(smoothed[1])))
            # print("Car Number: ", detection_number)
            # print("Left: ", left)
            # print("Right: ", right)
            # print("\n\n")
            cv2.rectangle(img, left, right, (255,0,255), 6)

        if labels[1] < self.detections:

            # print("We are in the dicey loop")
            for detection_number in range(labels[1], self.detections):
                # print("Car Number: ", detection_number)
                if(len(self.history[detection_number - 1]) > 1):
                    # print("history: ", self.history[detection_number - 1])
                    left, right = self.history[detection_number - 1]
                else:
                    # print("history: ", self.history[detection_number - 1])                    
                    left, right = self.history[detection_number - 1][0]
                # print("Left: ", left)
                # print("Right: ", right)
                # print("\n\n")
                cv2.rectangle(img, left, right, (255,0,255), 6)

        return img


    def process_image(self, frame):

        framebgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        features = extract_features(frame, self.cspace, self.orient, self.spatial_size, self.hist_bins, self.hist_range, self.pix, self.cell, self.hog_channel, self.path)

        x_start_stop = [576, 1280]
        y_start_stop = [400, 640]
        xy_window = [(96, 96), (128, 128), (160, 160)]
        xy_overlap = (0.5, 0.5)

        total_windows = []
        total_detections = []

        for window_size in xy_window:
        
            window_list = []
            window_list = slide_window(frame, x_start_stop, y_start_stop, window_size, xy_overlap)
            detections = search_windows(frame, window_list, self.classifier, self.X_scaler, color_space = self.cspace, hog_channel = self.hog_channel)
            total_windows.append(window_list)
            total_detections.append(detections)
            
        superimposed = np.copy(frame)
        heatmap = np.zeros_like(superimposed[:,:,0])

        for each_scale in total_detections:
            if len(each_scale) > 0:
                superimposed = draw_boxes(superimposed, each_scale)
                example = self.add_heat(heatmap, each_scale)
    
        thresholded_heatmap = self.threshold(heatmap, 2)
        labels = label(thresholded_heatmap)
        if labels[1] > self.detections:
            self.detections = labels[1]
    
        draw_img = self.draw_labeled_bboxes(np.copy(frame), labels)
        self.frames += 1

        return draw_img