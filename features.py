import cv2
import numpy as np 
from skimage.feature import hog


def bin_spatial(img, size = (32, 32)):

	features = cv2.resize(img, size).ravel()

	return features

def color_hist(img, nbins = 32, bins_range = (0, 256)):

    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features

def get_hog_features(img, orient = 9, pix_per_cell = 8, cell_per_block = 2, vis=False, feature_vec=True, hog_channel = 0):

    if vis == True:

        features, hog_image = hog(img, orientations=orient,
                          pixels_per_cell=(pix_per_cell, pix_per_cell), 
                          cells_per_block=(cell_per_block, cell_per_block), 
                          visualise=True, feature_vector=True,
                          block_norm="L2-Hys")
        return features, hog_image
    else:      
        features = hog(img, orientations=orient,
                          pixels_per_cell=(pix_per_cell, pix_per_cell), 
                          cells_per_block=(cell_per_block, cell_per_block), 
                          visualise=False, feature_vector=True,
                          block_norm="L2-Hys")

    return features

def extract_features(imgs, cspace='BGR', orient = 9, spatial_size=(32, 32), hist_bins=32,
						 hist_range=(0, 256), pix=8, cell=2, hog_channel=0, path=True):
    
    features = []
    
    if path == False:

        image = imgs
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
        if hog_channel == 'ALL':
            
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell = pix, cell_per_block = cell, 
                                vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features) 
            
        else:
            
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                    pix_per_cell = pix, cell_per_block = cell, vis=False, feature_vec=True)

        spatial_features = bin_spatial(feature_image, size=spatial_size)
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    
        return features

    else:

        for file in imgs:
            
            image = cv2.imread(file)

            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            else:
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell = pix, cell_per_block = cell, vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell=pix, cell_per_block=cell, vis=False, feature_vec=True)
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            features.append(np.concatenate((spatial_features, hist_features, hog_features)))

        return features
