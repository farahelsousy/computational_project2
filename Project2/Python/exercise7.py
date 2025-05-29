
import os
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog

from simulation_parameters import SimulationParameters
from util.run_open_loop import run_single
from util.entraining_signals import define_entraining_signals
from plotting_common import plot_time_histories

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]

def exercise7():

    pylog.info("Ex 7")
    log_path = './logs/exercise7/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    # Define the parameters for the simulation
    drive = 10  # drive signal amplitude
    cpg_amplitude_gain = np.array(
        [
            0.00824, 0.00328, 0.00328, 0.00370, 0.00451,
            0.00534, 0.00628, 0.00680, 0.00803, 0.01084,
            0.01115, 0.01149, 0.01655,
        ])
    weight = 2.0  # weight for the feedback signal
    n_iterations = 50001
    
    # Define the entraining signals
    entraining_signals = define_entraining_signals(
        n_iterations = n_iterations,
        frequency = 8.0,
        amplitude_degrees=45,
        n_joints=13,
        timestep=0.001,
        plot_signals=False,
    )

    pars = SimulationParameters(
        n_iterations=n_iterations,            
        timestep=0.001,
        controller='abstract oscillator',
        log_path=log_path,
        simulation_i=0,
        compute_metrics='all',
        print_metrics=True,
        return_network=True,
        feedback_weights_ipsi = weight,
        feedback_weights_contra = -weight,
        ws_ref = ws_ref,
        entraining_signals = entraining_signals,
        cpg_amplitude_gain = cpg_amplitude_gain,
        drive = drive,
    )

    # Run the simulation
    controller = run_single(pars)

if __name__ == '__main__':
    exercise7()

