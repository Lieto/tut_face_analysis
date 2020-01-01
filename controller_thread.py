import threading
import cv2
import time
import copy
import numpy as np

from unit_server import UnitServer
from grabber_thread import GrabberThread
from detection_thread import DetectionThread
from recognition_thread import RecognitionThread

class ControllerThread(threading.Thread):

    def __init__(self, config, logger):

        #super(threading.Thread, self).__init__()
        threading.Thread.__init__(self)

        #threading.Thread.__init__(self)

        self.logger = logger
        self.terminated = False
        self.caption = config["window_caption"]
        self.debug = config["debug"]
        self.initialize_fonts(config)

        self.min_detections = int(config["recognition_mindetections"])
        self.display_size = config["window_displaysize"]

        self.logger.info("Display size: {}".format(self.display_size))

        self.resolution = [1024, 768]

        queue_length = config["server_num_frames"]
        self.unit_server = UnitServer(queue_length)

        self.grabber_thread = GrabberThread(self, config=config, logger=self.logger)
        self.grabber_thread.start()

        self.faces = []
        self.detection_thread = DetectionThread(self, config, logger)
        self.detection_thread.start()

        self.recognition_thread = RecognitionThread(self, config, logger)
        self.recognition_thread.start()

        unused_width = self.resolution[0] - self.display_size[0]
        cv2.moveWindow(self.caption, unused_width // 2, 0)

        self.command_interface()

    def command_interface(self):

        while True:
            text = input("Enter command (Q)uit, (L)ist models, (S)witch model: ").lower()

            if text == "q":
                print("Bye!")
                self.terminate()
                break


    def initialize_fonts(self, config):

        self.free_type = None
        freetype_fontpath = config["window_freetype_fontpath"]
        sizetest_text = "FEMALE 100%"

        try:
            self.free_type = cv2.freetype.createFreeType2()
            self.free_type.loadFontData(fontFileName=freetype_fontpath, id=0)
            self.text_base_scale = 20
            self.text_base_width = self.free_type.getTextSize(sizetest_text, self.text_base_scale, -1)[0][0]

        except AttributeError:
            self.logger.error("OpenCV Freetype not found, falling back to standard OpenCV font...")
            self.text_base_scale = 0.6
            self.text_base_width = cv2.getTextSize(sizetest_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_base_scale, 2)[0][0]

    def run(self):

        while not self.terminated:
            time.sleep(0.5)

    def terminate(self):

        self.terminated = True

    def is_terminated(self):

        return self.terminated

    def put_unit(self, unit):

        self.show_video(unit)

        if not self.terminated:
            self.unit_server.put_unit(unit)

    def get_unit(self, caller, timestamp=None):

        return self.unit_server.get_unit(caller, timestamp)

    def show_video(self, unit):

        unit.acquire()
        frame = copy.deepcopy(unit.get_frame())
        unit.release()

        valid_faces = [f for f in self.faces if len(f['bboxes']) > self.min_detections]

        for face in valid_faces:
            self.draw_face(face, frame)

        frame = cv2.resize(frame, (self.display_size[0], self.display_size[1]))
        cv2.imshow(self.caption, frame)
        key = cv2.waitKey(10)

        if key == 27:
            self.terminate()

    def set_detections(self, detections, timestamps):

        for bbox, timestamp in zip(detections, timestamps):

            idx, dist = self.find_nearest_face(bbox)

            if dist is not None and dist < 50:

                self.faces[idx]['bboxes'].append(bbox)
                self.faces[idx]['timestamps'].append(timestamp)

                if len(self.faces[idx]['bboxes']) > 7:
                    self.faces[idx]['bboxes'].pop(0)
                    self.faces[idx]['timestamps'].pop(0)

            else:
                self.faces.append({'timestamps': [timestamp], 'bboxes': [bbox]})


        now = time.time()
        faces_to_remove = []

        for i, face in enumerate(self.faces):

            if now - face['timestamps'][-1] > 0.5:
                faces_to_remove.append(i)

        for i in faces_to_remove:

            try:
                self.faces.pop(i)
            except:
                pass

    def get_faces(self):

        if len(self.faces) == 0:
            return None
        else:
            return self.faces

    def find_nearest_face(self, bbox):

        distances = []

        x, y, w, h = bbox
        bboxCenter = [x + w / 2, y + h / 2]

        for face in self.faces:
            x, y, w, h = np.mean(face['bboxes'], axis=0)
            faceCenter = [x + w / 2, y + h / 2]

            distance = np.hypot(faceCenter[0] - bboxCenter[0],
                                faceCenter[1] - bboxCenter[1])

            distances.append(distance)

        if len(distances) == 0:
            minIdx = None
            minDistance = None
        else:
            minDistance = np.min(distances)
            minIdx = np.argmin(distances)

        return minIdx, minDistance

    def draw_face(self, face, img):

        bbox = np.mean(face['bboxes'], axis=0)

        self.draw_bounding_box(img, bbox)
        x, y, w, h = [int(c) for c in bbox]

        # 1. CELEBRITY TWIN

        celeb_identity = None

        # Clamp bounding box top to image
        y = 0 if y < 0 else y

        if "celebs" in face.keys():
            celeb_identity = self.AddCeleb(face, img, x, y, w, h)

        # Check if text can overlap the celeb texts (goes past the bounding box), if so decrease size
        text_size = self.text_base_scale

        if self.text_base_width > w:
            text_size *= w / self.text_base_width
            if self.free_type:
                text_size = int(text_size)  # Freetype doesn't accept float text size.

        # 1. AGE

        if "age" in face.keys():
            age = face['age']
            annotation = "Age: %.0f" % age
            txtLoc = (x, y + h + 30)
            self.write_text(img, annotation, txtLoc, text_size)

            # 2. GENDER

        if "gender" in face.keys():
            gender = "MALE" if face['gender'] > 0.5 else "FEMALE"
            genderProb = max(face["gender"], 1 - face["gender"])
            annotation = "%s %.0f %%" % (gender, 100.0 * genderProb)
            txtLoc = (x, y + h + 60)
            self.write_text(img, annotation, txtLoc, text_size)

        # 3. EXPRESSION

        if "expression" in face.keys():
            expression = face["expression"]
            annotation = "%s" % (expression)
            txtLoc = (x, y + h + 90)
            self.write_text(img, annotation, txtLoc, text_size)

        if celeb_identity:
            annotation = "CELEBRITY"
            txtLoc = (x + w, y + h + 30)
            self.write_text(img, annotation, txtLoc, text_size)

            annotation = "TWIN"  # (%.0f %%)" % (100*np.exp(-face["celeb_distance"]))
            txtLoc = (x + w, y + h + 60)
            self.write_text(img, annotation, txtLoc, text_size)

            annotation = celeb_identity
            txtLoc = (x + w, y + h + 90)
            self.write_text(img, annotation, txtLoc, text_size)

        # DEBUG ONLY - Visualize aligned face crop in corner.
        if self.debug and "crop" in face.keys():
            crop = face["crop"]
            crop = cv2.resize(crop, (100, 100))
            croph, cropw = crop.shape[0:2]
            imgh, imgw = img.shape[0:2]

            img[:croph, imgw - cropw:, :] = crop[..., ::-1]

    def write_text(self, img, annotation, location, size):
        if self.free_type:
            self.free_type.putText(img=img,
                            text=annotation,
                            org=location,
                            fontHeight=size,
                            color=(255, 255, 0),
                            thickness=-1,
                            line_type=cv2.LINE_AA,
                            bottomLeftOrigin=True)
        else:
            annotation = annotation.replace('ä', 'a').replace('ö', 'o').replace('å', 'o')
            cv2.putText(img,
                        text=annotation,
                        org=location,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=size,
                        color=[255, 255, 0],
                        thickness=2)

    def draw_bounding_box(self, img, bbox):

        x, y, w, h = [int(c) for c in bbox]

        m = 0.2

        # Upper left corner
        pt1 = (x, y)
        pt2 = (int(x + m * w), y)
        cv2.line(img, pt1, pt2, color=[255, 255, 0], thickness=2)

        pt1 = (x, y)
        pt2 = (x, int(y + m * h))
        cv2.line(img, pt1, pt2, color=[255, 255, 0], thickness=2)

        # Upper right corner
        pt1 = (x + w, y)
        pt2 = (x + w, int(y + m * h))
        cv2.line(img, pt1, pt2, color=[255, 255, 0], thickness=2)

        pt1 = (x + w, y)
        pt2 = (int(x + w - m * w), y)
        cv2.line(img, pt1, pt2, color=[255, 255, 0], thickness=2)

        # Lower left corner
        pt1 = (x, y + h)
        pt2 = (x, int(y + h - m * h))
        cv2.line(img, pt1, pt2, color=[255, 255, 0], thickness=2)

        pt1 = (x, y + h)
        pt2 = (int(x + m * w), y + h)
        cv2.line(img, pt1, pt2, color=[255, 255, 0], thickness=2)

        # Lower right corner
        pt1 = (x + w, y + h)
        pt2 = (x + w, int(y + h - m * h))
        cv2.line(img, pt1, pt2, color=[255, 255, 0], thickness=2)

        pt1 = (x + w, y + h)
        pt2 = (int(x + w - m * w), y + h)
        cv2.line(img, pt1, pt2, color=[255, 255, 0], thickness=2)









