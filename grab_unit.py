import time

class GrabUnit:

    def __init__(self, logger, frame):

        self.logger = logger
        self.timestamp = time.time()
        self.detected = False
        self.age_recognized = False
        self.gender_recognized = False
        self.expression_recognized = False

        self.processes = 0

        self.frame = frame

    def get_timestamp(self):

        return self.timestamp

    def get_frame(self):

        return self.frame

    def acquire(self):

        self.processes += 1

    def release(self):

        self.processes -= 1

    def is_free(self):

        if self.processes == 0:
            return True
        else:
            return False

    def get_num_processes(self):

        return self.processes

    def get_timestamp(self):

        return self.timestamp

    def get_age(self):

        return time.time() - self.timestamp

    def set_detected(self):

        self.detected = True

    def set_age_recognized(self):

        self.age_recognized = True

    def set_gender_recognized(self):

        self.gender_recognized = True

    def set_expression_recognized(self):

        self.expression_recognized = True

    def is_detected(self):

        return self.detected

    def is_age_recognized(self):

        return self.age_recognized

    def is_gender_recognized(self):

        return self.gender_recognized

    def is_expression_recognized(self):

        return self.expression_recognized
