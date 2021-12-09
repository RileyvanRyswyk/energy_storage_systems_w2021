

class Transaction:

    def __init__(self, start_time, duration, power):
        self.start_time = start_time
        self.end_time = start_time + duration
        self.power = power

    def __contains__(self, t):
        return self.start_time <= t <= self.end_time
