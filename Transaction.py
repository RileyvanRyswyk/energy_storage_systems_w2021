

class Transaction:

    def __init__(self, base_start_time, start_time, duration, power):
        self.base_start_time = base_start_time  # market data time
        self.start_time = start_time            # simulation time
        self.end_time = start_time + duration
        self.power = power

    def __contains__(self, t):
        return self.start_time <= t < self.end_time
