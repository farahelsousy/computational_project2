
import os
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog

from util.rw import load_object
from util.run_open_loop import run_multiple
from util.entraining_signals import define_entraining_signals
from simulation_parameters import SimulationParameters
from plotting_common import plot_2d, save_figures

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]

def exercise8():

    pylog.info("Ex 8")
    log_path = './logs/exercise8/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    n_freqs = 10
    n_weights = 5
    n_iterations=n_iterations = 10001 
    timestep=timestep = 0.001

    pars_list = [
            SimulationParameters(
                simulation_i = i*n_freqs+j,
                controller="abstract oscillator",
                compute_metrics="all",
                n_iterations=n_iterations,
                timestep=timestep,
                headless=True,
                video_record=False,
                log_path=log_path,
                return_network=False,
                print_metrics=False,
                feedback_weights_ipsi = -weight,
                feedback_weights_contra = weight,
                ws_ref = ws_ref,
                entraining_signals = define_entraining_signals(
                    n_iterations = n_iterations,
                    frequency = frequency,
                    timestep=timestep)
                )
            for i, weight in enumerate(np.linspace(0, 2, n_weights))
            for j, frequency in enumerate(np.linspace(3.5, 10, n_freqs))      
        ]
    
    # Run the simulation
    controllers = run_multiple(pars_list, num_process=6)

if __name__ == '__main__':

    exercise8()

