import cv2
import threading
import time

class DetectionThread(threading.Thread):

    def __init__(self, parent, config, logger):

        threading.Thread.__init__(self)

        self.logger = logger

        self.logger.info("Intializing detection thread...")
        self.parent = parent

        frozen_graph = config["detection_inference_graph"]
        text_graph = config["detection_text_graph"]

        self.cv_net = cv2.dnn.readNetFromTensorflow(frozen_graph, text_graph)

        self.width = config["detection_input_width"]
        self.height = config["detection_input_height"]

    def run(self):

        while self.parent.is_terminated() == False:

            unit = None

            while unit == None:

                unit = self.parent.get_unit(self)

                if unit == None:
                    time.sleep(0.1)

                if self.parent.is_terminated():
                    break
            if self.parent.is_terminated():
                break

            img = unit.get_frame()

            detection_img = img.copy()
            unit.release()

            rows, cols = img.shape[0:2]
            self.cv_net.setInput(cv2.dnn.blobFromImage(detection_img, size=(self.width, self.height),
                                                       swapRB=True, crop=False))

            timer = time.time()
            cv_out = self.cv_net.forward()

            bboxes = []
            timestamps = []

            for detection in cv_out[0, 0, :, :]:
                score = float(detection[2])

                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)
                width = right - left
                height = bottom - top

                if score > 0.3 and width > 60:
                    bboxes.append([left, top, width, height])
                    timestamps.append(unit.get_timestamp())

            self.parent.set_detections(bboxes, timestamps)

