
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

    
    # Define the entraining signals
    entraining_signals = define_entraining_signals(
        n_iterations = 50001,
        frequency = 8.0,
        amplitude_degrees=45,
        n_joints=13,
        timestep=0.001,
        plot_signals=False,
    )


    pars = SimulationParameters(
        n_iterations=50001,             # 10 s @ 0.001 s timestep
        timestep=0.001,
        controller='abstract oscillator',
        log_path=log_path,
        simulation_i=0,
        compute_metrics='all',
        print_metrics=True,
        return_network=True,
        feedback_weights_ipsi=-2,
        feedback_weights_contra=2,
        ws_ref=ws_ref,
        entraining_signals=entraining_signals
    )

    # Run the simulation
    controller = run_single(pars)



if __name__ == '__main__':

    exercise7()

