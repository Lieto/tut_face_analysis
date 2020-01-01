import threading
from detection_thread import DetectionThread
from recognition_thread import RecognitionThread

class UnitServer:

    def __init__(self, logger, max_units=4):

        self.logger = logger
        self.max_units = max_units
        self.units = []
        self.mutex = threading.Lock()

    def get_unit(self, caller, timestamp=None):

        self.mutex.acquire()

        unit = None

        if timestamp is not None:

            for f in self.units:
                if f.get_timestamp() == timestamp:
                    unit = f
        else:

            if isinstance(caller, DetectionThread):

                valid_units = [f for f in self.units if f.is_detected() == False]

                if len(valid_units) == 0:
                    unit = None
                else:
                    unit = valid_units[-1]
                    unit.acquire()
                    unit.set_detected()

            if isinstance(caller, RecognitionThread):

                valid_units = [f for f in self.units if f.is_detected() == True and f.is_age_recognized() == False]

                if len(valid_units) == 0:
                    unit = None
                else:
                    unit = valid_units[-1]
                    unit.acquire()
                    unit.set_detected()

        self.mutex.release()

        return unit

    def put_unit(self, unit):

        self.mutex.acquire()

        if len(self.units) >= self.max_units:

            if self.units[0].is_free():
                self.units.pop(0)

        if len(self.units) < self.max_units:
            self.units.append(unit)
        else:
            pass

        self.mutex.release()
