
import os
import farms_pylog as pylog
import numpy as np
import matplotlib.pyplot as plt

from util.rw import load_object
from util.run_closed_loop import run_multiple
from simulation_parameters import SimulationParameters
from plotting_common import plot_1d, save_figures

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]


def exercise6():

    pylog.info("Ex 6")
    pylog.info("Implement exercise 6")
    log_path = './logs/exercise6/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)


if __name__ == '__main__':

    exercise6()

