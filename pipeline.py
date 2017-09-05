from moviepy.editor import VideoFileClip
import pickle
from scipy.ndimage.measurements import label
from helper_functions import *


class Tracker(object):
    def __init__(self):
        self.cumheat = []
        self.input_video_clip = VideoFileClip(video_input)
        self.output_video_clip = self.input_video_clip.fl(self.pipeline)

    def pipeline(self, gf, t):
        img = gf(t)
        aux = np.copy(img)
        img = img.astype(np.float32) / 255  # im you're searching is a jpg (0 to 255), trained on .png (0 to 1 by mpimg)
        windows = slide_variable_window(img, x_start, y_start, xy_window, x_overlap, y_overlap)

        hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space, spatial_size=spatial_size,
                                     hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = add_heat(heat, hot_windows)  # Add heat to each box in box list
        self.cumheat.append(heat)
        heat = apply_threshold(sum(self.cumheat[-n_frames:]), heat_thres)  # Apply thresh to help remove false positives
        heatmap = np.clip(heat, 0, 255)  # Visualize the heatmap when displaying

        labels = label(heatmap)  # Find final boxes from heatmap using label function
        draw_img = draw_labeled_bboxes(aux, labels)

        return draw_img


with open('pickle_files/model_and_pars.pickle', 'rb') as f:
    svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, \
    spatial_feat, hist_feat, hog_feat = pickle.load(f)
with open('pickle_files/window_search_pars.pickle', 'rb') as f:
    x_start, y_start, x_overlap, y_overlap, xy_window = pickle.load(f)

n_frames = 35
heat_thres = 28
video_input = './project_video.mp4'
video_output = './project_video_output.mp4'

result = Tracker()
result.output_video_clip.write_videofile(video_output, audio=False)
