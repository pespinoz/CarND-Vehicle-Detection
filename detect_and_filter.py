import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import pickle
import glob
from helper_functions import *

"""
This file computes bounding boxes for detected vehicles and filter false positives in a time-series set of images 
(seven ims extracted from the project video using `ffmpeg`)

"""

# We begin by loading the model and feature extraction parameters
with open('pickle_files/model_and_pars.pickle', 'rb') as f:
    svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, \
    spatial_feat, hist_feat, hog_feat = pickle.load(f)

#######################################################################################################################

# Parameters
x_start = 200       # x pixel start; previous value: 200
y_start = 375       # y pixel start; previous value: 375
x_overlap = 0.9     # x overlap; previous value: 0.7
y_overlap = 0.7     # y overlap; previous value: 0.7
# List of xy_window sizes:
xy_window = [(64, 64), (80, 80), (96, 96), (116, 116), (132, 132), (132, 132), (132, 132), (132, 132), (132, 132),
             (132, 132)]  # previous value = [(96, 96)]*20

images = sorted(glob.glob('test_images/ffmpeg_extracted_images/*.jpg'))
heatcum = []        # Heat cumulative map
for image in images:
    img = mpimg.imread(image)
    aux = np.copy(img)
    img = img.astype(np.float32)/255

    # Generates variable size windows (in the y-axis):
    windows = slide_variable_window(img, x_start, y_start,  xy_window, x_overlap=x_overlap, y_overlap=y_overlap)

    # Run the classifier in each window:
    hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space, spatial_size=spatial_size,
                                 hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    # Annotate bounding boxes in the windows where the classifier detects a vehicle:
    window_img = draw_boxes(aux, hot_windows, color=(0, 0, 255), thick=6)

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, hot_windows)              # Add heat to each box in box list
    heatmap1 = np.clip(heat, 0, 255)                # Visualize the heatmap when displaying

    heatcum.append(heat)
    # Generates the bounding box and heatmap plots in the extracted images from the video:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.imshow(window_img)
    ax1.set_title(image.split('/')[-1], fontsize=15)
    ax2.imshow(heatmap1, cmap='hot')
    ax2.set_title('Heatmap ' + image.split('/')[-1], fontsize=15)
    fig.savefig('output_images/' + image.split('/')[-1], dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)
    plt.close(fig)


img = mpimg.imread(images[-1])
aux = np.copy(img)
heatmap = apply_threshold(sum(heatcum), 3)         # Apply threshold to help remove false positives
labels = label(heatmap)                            # Find final boxes from heatmap using label function
draw_img = draw_labeled_bboxes(aux, labels)

# Generates the final plot:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.imshow(labels[0])
ax1.set_title('Integrated Heatmap', fontsize=15)
ax2.imshow(draw_img, cmap='hot')
ax2.set_title('Last frame in series', fontsize=15)
fig.savefig('output_images/final.jpg', dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
            papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, frameon=None)

with open('pickle_files/window_search_pars.pickle', 'wb') as f:
    pickle.dump([x_start, y_start, x_overlap, y_overlap, xy_window], f)
