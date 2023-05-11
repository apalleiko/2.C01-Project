import numpy as np
import cv2


class kinem:
    def __init__(self, frames, img_x, img_y, length_of_video):
        self.img_x = img_x
        self.img_y = img_y
        self.tot_area = img_x * img_y
        self.frame_to_data = dict()
        self.tot_frames = frames
        self.vid_len = length_of_video
        self.fps = self.tot_frames / length_of_video

        self.pix_to_known = None

        for frame in range(frames):
            self.frame_to_data[frame] = (0, (0, 0))

    def set_known_distance_golfball(self, pix_dist, known_dist):
        # golf ball diameter known: 4.3 cm or 0.043 m
        # 180
        self.pix_to_known = known_dist / pix_dist

    def frame_to_mask_data(self, frame, mask):
        # recast int
        mask_int = mask.astype('uint8')

        contour, _ = cv2.findContours(mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # moments
        M = cv2.moments(mask_int)
        # print(frame)
        # # calculate x,y coordinate of center
        # if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])


        if not self.pix_to_known:
            midline = [i for i, x in enumerate(mask_int[cX]) if x == 1]
            diameter = midline[-1] - midline[0]
            self.set_known_distance_golfball(diameter, 0.043)

        area = cv2.contourArea(contour[0])
        if cX:
            self.frame_to_data[frame] = (area, (cX, cY))

    def get_velocities(self):
        start = self.frame_to_data[0][1]
        self.vels = []
        for i in range(1, len(self.frame_to_data.keys())-1):
            temp = self.frame_to_data[i][1]
            self.vels.append(np.array((temp[0] - start[0], temp[1] - start[1]))*self.pix_to_known / self.fps)
            start = temp
        return self.vels
