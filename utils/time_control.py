import time
import datetime


class TimeControl:
    def __init__(self):
        self.now_time = None

    @staticmethod
    def now_day_str():
        curr_day = datetime.datetime.now()
        curr_day_str = '{}-{}-{}'.format(curr_day.year, curr_day.month, curr_day.day)
        return curr_day_str

    @staticmethod
    def now_time_str():
        curr_time = datetime.datetime.now()
        curr_time_str = '{}-{}-{}_{}.{}.{}'.format(curr_time.year, curr_time.month, curr_time.day,
                                                   curr_time.hour, curr_time.minute, curr_time.second)
        return curr_time_str

    def timer_start(self):
        self.now_time = time.time()

    def timer_end(self):
        duration_time = time.time() - self.now_time
        return str(datetime.timedelta(seconds=duration_time))

    def timer_log(self):
        duration_time = time.time() - self.now_time
        self.timer_start()
        return str(datetime.timedelta(seconds=duration_time))
