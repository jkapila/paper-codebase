"""
Adaptor class for video loading, streaming, parsing and conversion
"""

import cv2
import skvideo
from utils.Utils import resizing

class VideoCamera(object):
    def __init__(self, videostream):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        try:
            self.video = cv2.VideoCapture(videostream)
            self.lib = "opencv"
        except:
            inputparameters = {}
            outputparameters = {}
            self.video = skvideo.io.FFmpegReader(videostream,
                                             inputdict=inputparameters,
                                             outputdict=outputparameters)
            # self.video = skvideo.io.vreadr(videostream)
            self.lib = "skvideo"
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        self.resize = False
        print('Video Library used: ', self.lib)

    def __del__(self):
        if self.lib == "opencv":
            self.video.release()
        elif self.lib == "skvideo":
            self.video.close()

    def resize_image(self, resize_image_px=500):
        self.resize = True
        self.resize_pixel_length = resize_image_px

    def has_frame(self):
        if self.lib == "opencv":
            return self.video.isOpened()
        elif self.lib == "skvideo":
            pass

    def __len__(self):
        if self.lib == "opencv":
            return self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        elif self.lib == "skvideo":
            return self.video.getShape()[0]

    def get_frame(self):
        if self.lib == "opencv":
            success, image = self.video.read()
        elif self.lib == "skvideo":
            image = self.video.nextFrame()

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        # ret, jpeg = cv2.imencode('.jpg', image)

        if self.resize:
            image = resizing(image, self.resize_pixel_length)

        return image

if __name__ == '__main__':
    cam = VideoCamera('D:\innovate\data\VID_KUMARJIT.mp4')
    cam.resize_image()
    # print('Length of stream', len(cam))

    while cam.has_frame():
        cv2.imshow("image",cam.get_frame())