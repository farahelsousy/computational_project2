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

def plot_E8(n_weights, n_freqs, logdir):
    # Plot the results of the simulations in exercise 1.4 (2.4)
    #   n_weights = number of weights
    #   n_twl = number of TWL
    #   start_idx = index of the first controller to load
    #   logdir = directory where the simulation files are stored
    #   title = title of the plot for saving

    # For each twl-value make a matrix with the speed and CoT as a function of frequnecies
    mat_diff_actual = np.zeros((n_freqs,n_weights))

    # Define reference controller
    ref_controller = load_object(logdir+"controller"+str(0))
    if ref_controller.pars.feedback_weights_ipsi != 0:
        raise Exception("Error: Please include w=0.")


    mat_diff_entrain = np.linspace(3.5, 10, n_freqs) - np.mean(ref_controller.metrics["neur_frequency"])
    mat_w = np.linspace(0, 2, n_weights)

    title = "entraining_signals"

    for i in range(n_freqs):
        for j in range(n_weights):
            # load controller
            controller = load_object(logdir+"controller"+str(j*n_freqs+i))

            # Get metrics
            mat_diff_actual[i,j] = np.mean(controller.metrics["neur_frequency"]) - np.mean(ref_controller.metrics["neur_frequency"])

    # Create an array of text containing legends for the twl-values
    w_legends = [f"w = {value:.2f}" for value in mat_w]

    # Plot the speed and frequency matrices in the same plot. Set x-axis as frequency and y-axis as fwd speed.
    plt.figure(title, figsize=(15, 8))
    plt.plot(mat_diff_entrain, mat_diff_actual, marker='o', markersize=5, label=w_legends)

    # Define plot properties
    plt.xlabel("Entrainment freq. - reference freq. [Hz]")
    plt.ylabel("Actual freq. - reference freq. [Hz]")
    plt.legend(loc='center left',bbox_to_anchor=(1.0, 0.5),fontsize='small',)
    plt.minorticks_on()
    plt.grid(which='major', color='darkgrey', linestyle='-', linewidth=0.75)
    plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)

    plt.savefig(os.path.join(logdir, 'entraining_signals.png'), dpi=600, bbox_inches='tight')

def exercise8(run_sim = True, run_plot = True):

    pylog.info("Ex 8")
    log_path = './logs/exercise8/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    # Define the parameters for the simulation
    n_freqs = 51
    n_weights = 11   
    n_iterations = 50001 
    timestep=timestep = 0.001
    drive = 10
    cpg_amplitude_gain = np.array(
        [
            0.00824, 0.00328, 0.00328, 0.00370, 0.00451,
            0.00534, 0.00628, 0.00680, 0.00803, 0.01084,
            0.01115, 0.01149, 0.01655,
        ])

    if run_sim:
        pars_list = [
                SimulationParameters(
                    simulation_i = j*n_freqs+i,
                    controller="abstract oscillator",
                    compute_metrics="all",
                    n_iterations=n_iterations,
                    timestep=timestep,
                    headless=True,
                    video_record=False,
                    log_path=log_path,
                    return_network=False,
                    print_metrics=False,
                    feedback_weights_ipsi = weight,
                    feedback_weights_contra = -weight,
                    ws_ref = ws_ref,
                    drive = drive,
                    cpg_amplitude_gain = cpg_amplitude_gain,
                    entraining_signals = define_entraining_signals(
                        n_iterations = n_iterations,
                        frequency = frequency,
                        timestep=timestep)
                    )
                for i, frequency in enumerate(np.linspace(3.5, 10, n_freqs))
                for j, weight in enumerate(np.linspace(0, 2, n_weights))
            ]
        
        # Run the simulation
        controllers = run_multiple(pars_list, num_process=6)

    if run_plot:
        plot_E8(n_weights, n_freqs, log_path)

if __name__ == '__main__':
    run_sim = False
    run_plot = True
    exercise8(run_sim, run_plot)

