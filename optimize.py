import json
import math
from datetime import timedelta
from multiprocessing import cpu_count, Pool, shared_memory

import numpy as np
import pandas as pd
import tqdm

from Battery import Battery
from FrequencyData import FrequencyData
from MarketData import MarketData
import StorageSystem
import Plotter

DEV = False     # limit data for faster processing
DT = 5          # seconds


# optimize configurations
def optimize_configurations():
    systems = create_configurations()
    # valid_systems = validate_configurations(systems)

    # battery = Battery(eta_char=0.90, eta_disc=0.90, eta_self_disc=0, capacity_nominal=7.5)
    # ss = StorageSystem.StorageSystem(battery=battery, p_market=5, p_max=6.25, soc_target=0.5)

    results, financials = run_economic_simulation(systems)

    Plotter.plot_revenues(results, financials, save_fig=True, path='figures/')

    print(financials)


def create_configurations():
    p_market = 5
    p_max = 1.25 * p_market

    # compute the minimum storage size in 0.25 MWh increments
    weighted_duration = StorageSystem.ALERT_DURATION
    weighted_duration += StorageSystem.TRANSACTION_DELAY * StorageSystem.P_MAX_CONST_DEV
    e_min = 2 * p_market * weighted_duration
    e_min = math.ceil(e_min * 4) / 4.0

    capacities = np.arange(e_min, 1.5*e_min, 0.25)
    soc_targets = [0.5]     # doesn't make a huge difference
    triggers = np.arange(0.1, 0.3, 0.05)
    systems = []
    for capacity in capacities:
        for soc_target in soc_targets:
            for trigger in triggers:
                battery = Battery(eta_char=0.90, eta_disc=0.90, eta_self_disc=0, capacity_nominal=capacity)
                ss = StorageSystem.StorageSystem(
                    battery=battery,
                    p_market=p_market,
                    p_max=p_max,
                    soc_target=soc_target,
                    sell_trigger=0.5 + trigger,
                    buy_trigger=0.5 - trigger
                )

                # avoid simulating duplicate systems, due to force parameter modifications
                # in order to meet basic requirements
                if ss not in systems:
                    systems.append(ss)

    return systems


#  Run configurations against the prequalification (PQ) data
#  and verify that the min or max state of charge was never reached,
#  since the system never enters the alert state with this dataset
def validate_configurations(systems):
    print("Validating {} configurations.".format(len(systems)))

    duration = None
    if DEV:
        duration = timedelta(days=2)

    results = run_parallel_simulations(
        data_src=FrequencyData.PQ_DATA,
        duration=duration,
        dt=DT,
        storage_systems=systems,
        plot_path='figures/validation/'
    )

    valid_systems = []
    for index, result in enumerate(results):
        if np.min(result.sim_data['batt_soc']) > result.soc_min \
         and np.max(result.sim_data['batt_soc']) < result.soc_max:
            valid_systems.append(result.reset())

    print("{} systems are valid.".format(len(valid_systems)))
    return valid_systems


def run_economic_simulation(systems):
    print("Evaluating economics for {} configurations.".format(len(systems)))

    duration = None
    if DEV:
        duration = timedelta(days=2)

    # market data is too small to use shared memory
    md = MarketData()
    for system in systems:
        system.link_market_data(md)

    results = run_parallel_simulations(
        data_src='data/202111_Frequenz.csv',
        duration=duration,
        dt=DT,
        storage_systems=systems,
        plot_path='figures/economic/'
    )

    financials = list(map(lambda ss: ss.compute_financials(), results))

    return results, financials


def run_parallel_simulations(data_src, duration, dt, storage_systems, plot_path=None):
    shm_links, freq = load_sharable_block(data_src=data_src, duration=duration, dt=dt)

    systems = map(lambda ss: (freq, dt, ss, plot_path), storage_systems)

    try:
        with Pool(processes=min(8, cpu_count())) as pool:
            results = []
            for result in tqdm.tqdm(
                pool.imap_unordered(simulate_storage_config_star, systems),
                total=len(storage_systems)
            ):
                results.append(result)

            # res = pool.starmap_async(simulate_storage_config, systems, callback=log_result)
            # print("Final result: ", res.get())
    except Exception as err:
        for shm_link in shm_links:
            shm_link.close()
            shm_link.unlink()

        raise f"Unexpected {err=}, {type(err)=}"

    return results


def log_result(result):
    print("Succesfully get callback! With result: ", result)


def load_sharable_block(data_src, duration, dt):
    fd = FrequencyData(data_src)
    data = fd.get_data_subset(duration=duration, dt=dt)
    index = data.index.to_numpy()
    freq = data['freq'].to_numpy()

    index_shm, index_arr = create_shared_block(index, 'datetime64[s]')
    freq_shm, _ = create_shared_block(freq, np.float32)

    freq_info = {'size': index_arr.size, 'index': index_shm.name, 'freq': freq_shm.name}
    shm_links = [index_shm, freq_shm]

    # clean up memory
    del index, freq, data, fd

    return shm_links, freq_info


def create_shared_block(src_array, dtype):
    shm = shared_memory.SharedMemory(create=True, size=src_array.nbytes)
    # Now create a NumPy array backed by shared memory
    np_array = np.ndarray(src_array.shape, dtype=dtype, buffer=shm.buf)
    np_array[:] = src_array[:]  # Copy the original data into shared memory
    return shm, np_array


def simulate_storage_config_star(args):
    return simulate_storage_config(*args)


def simulate_storage_config(freq_info, dt, ss, plot_path=None):

    index_shm = shared_memory.SharedMemory(name=freq_info['index'])
    freq_shm = shared_memory.SharedMemory(name=freq_info['freq'])
    index = np.ndarray((freq_info['size'],), dtype='datetime64[s]', buffer=index_shm.buf)
    freq = np.ndarray((freq_info['size'],), dtype=np.float32, buffer=freq_shm.buf)

    ss.init_sim_data(freq.size)
    current_day = None
    for i in range(0, freq.size):
        t1 = pd.Timestamp(index[i])
        ss.execute_step(freq=freq[i], dt=dt, t=t1)
        if current_day is None or current_day != t1.date().day:
            current_day = t1.date().day
            # print("Currently evaluating {}".format(t1.date()))

    if plot_path is not None:
        Plotter.plot_time_curves(ss, save_fig=True, path=plot_path)
        Plotter.plot_rel_freq_data(ss, save_fig=True, path=plot_path)

    index_shm.close()
    freq_shm.close()

    return ss
