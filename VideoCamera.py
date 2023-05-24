import cv2
class VideoCamera(object):
    def __init__(self, index):
        self.video = cv2.VideoCapture(index)

    # def __del__(self):
    #     self.video.release()

    def videoRead(self):
        ret, frame = self.video.read()
        return ret, frame
        