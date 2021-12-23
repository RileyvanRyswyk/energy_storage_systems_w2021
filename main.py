from Battery import Battery
from FrequencyData import FrequencyData
from StorageSystem import StorageSystem
import Plotter
import optimize


def simulate_storage(duration):
    fd = FrequencyData("data/202101_Frequenz.csv")
    dt = 5  # seconds
    data = fd.get_data_subset(duration=duration, dt=dt)

    # https://www.mdpi.com/2313-0105/2/3/29/pdf section 5.1
    # from Energy Neighbour project, self discharge is negligible during constant operation
    # losses primarily from power electronics, auxillary equipment
    battery = Battery(eta_char=0.90, eta_disc=0.90, eta_self_disc=0, capacity_nominal=7.5)
    ss = StorageSystem(battery, p_market=5, p_max=6.25, soc_target=0.6)

    ss.init_sim_data(len(data))
    current_day = None
    for row in data.itertuples(index=True):
        ss.execute_step(freq=row.freq, dt=dt, t=row.Index)
        if current_day is None or current_day != row.Index.date().day:
            current_day = row.Index.date().day
            print("Currently evaluating {}".format(row.Index.date()))

    print("Equivalent full cycle count: {}".format(ss.battery.eq_full_cycle_count))
    print("End SOC: {:.2%}".format(ss.battery.soc))
    print("End Energy Gain: {:.2f} MWh".format((ss.battery.soc-ss.battery.starting_soc) * ss.battery.capacity_nominal))

    Plotter.plot_time_curves(ss)
    Plotter.plot_rel_freq_data(ss)


if __name__ == "__main__":
    # simulate_storage(duration=timedelta(days=1))
    optimize.optimize_configurations()
