

class Battery:

    # eta_char [%/100]
    # eta_disc [%/100]
    # eta_self_disc [%/100 per day]
    def __init__(self, soc=0.5, eta_char=1, eta_disc=1, eta_self_disc=0):
        self.soc = soc                                          # state of charge [%/100]
        self.eta_char = eta_char                                # charging efficiency [%/100]
        self.eta_disc = eta_disc                                # discharging efficiency [%/100]
        self.eta_self_disc = eta_self_disc / (3600 * 24)        # self discharge [%/100 per s]
        self.cycle_count = 0                                    # equivalent full cycle count

    def charge(self, energy):
        net_energy = energy * self.eta_char
        self.soc += net_energy
        self.cycle_count += 0.5 * net_energy

    def discharge(self, energy):
        req_energy = energy / self.eta_disc
        self.soc -= req_energy
        self.cycle_count += 0.5 * req_energy

    # dt : time delta [s]
    def count_self_discharge_losses(self, dt):
        self.soc -= dt * self.eta_self_disc
        pass
