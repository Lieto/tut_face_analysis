import threading
import cv2
import time

#from controller_thread import ControllerThread
from grab_unit import GrabUnit

class GrabberThread(threading.Thread):

    def __init__(self, parent, config, logger):

        #super(threading.Thread, self).__init__()
        threading.Thread.__init__(self)

        self.logger = logger
        cam_id = config["camera_id"]

        cam_resolution = config["camera_resolution"]
        self.logger.info("Using camera {} at resolution {}x{}".format(cam_id, cam_resolution[0], cam_resolution[1]))

        self.flip_hor = config["camera_flip_horizontal"]

        self.video = cv2.VideoCapture("test_4.mp4")
        #self.video = cv2.VideoCapture(cam_id)
        self.parent = parent

        self.logger.info("Grabber Thread initialized...")

    def run(self):

        time.sleep(10.0)

        while not self.parent.is_terminated():

            stat, frame = self.video.read()

            if frame is not None and not self.parent.is_terminated():

                if self.flip_hor:
                    frame = frame[:, ::-1, ...]

                unit = GrabUnit(self.logger, frame)

                self.parent.put_unit(unit)
