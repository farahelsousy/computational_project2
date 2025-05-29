import os
import farms_pylog as pylog
import numpy as np
import matplotlib.pyplot as plt

from util.rw import load_object
from util.run_closed_loop import run_multiple
from simulation_parameters import SimulationParameters
from plotting_common import plot_1d, save_figures
from util.zebrafish_hyperparameters import define_hyperparameters

# Define general parameters for the simulation
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]

num_process = 4 # number of processes to run the simulation in parallel

def plot_E6(n_weights, logdir, title):
    # Plot the results of the simulations in exercise 1.4 (2.4)
    #   n_weights = number of weights
    #   n_twl = number of TWL
    #   start_idx = index of the first controller to load
    #   logdir = directory where the simulation files are stored
    #   title = title of the plot for saving

    # For each twl-value make a matrix with the speed and CoT as a function of frequnecies
    mat_weights = np.zeros((n_weights*2,2))

    axis_List = ["Neural Frequency", "Neural TWL", "Cost of Transport", "Energy Consumption", "Forward speed","Mean joint amplitudes", "Sum of torques"]
    metrics_list= ["neur_frequency", "neur_twl", "mech_cot", "mech_energy", "mech_speed_fwd","mech_joint_amplitudes", "mech_torque"]
    n_metrics = 7
    mat_metrics = np.zeros((n_weights*2,n_metrics))
    mat_joint_amplitudes = np.zeros((n_weights*2,15))  # Special case, as we want to plot the amplitude for each joint separately. There are 15 joints in tota.

    for i in range(n_weights*2):
        # load controller
        controller = load_object(logdir+"controller"+str(i))

        # Get weights
        mat_weights[i,0] = controller.pars.feedback_weights_ipsi
        mat_weights[i,1] = controller.pars.feedback_weights_contra

        # Get metrics
        for j in range(n_metrics):
            mat_metrics[i,j] = np.mean(controller.metrics[metrics_list[j]])

        # Special case for joint amplitudes
        mat_joint_amplitudes[i,:] = controller.metrics["mech_joint_amplitudes"]

    for j in range(n_metrics):
        # Plot the speed and frequency matrices in the same plot. Set x-axis as frequency and y-axis as fwd speed.
        plt.figure(metrics_list[j], figsize=[10, 10])
        plt.plot(mat_weights[:n_weights,0], mat_metrics[:n_weights,j], marker='o', markersize=5)
        plt.plot(mat_weights[n_weights:2*n_weights,1], mat_metrics[n_weights:2*n_weights,j], marker='o', markersize=5)
        plt.legend(["$w^{ipsi}$ sweep", "$w^{contra}$ sweep"], 
                loc='upper right',
                fontsize='small',
                )
        
        # Define plot properties
        plt.xlabel("Weight")
        plt.ylabel(axis_List[j])
        plt.minorticks_on()
        plt.grid(which='major', color='darkgrey', linestyle='-', linewidth=0.75)
        plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)

        save_figures(
            logdir=logdir,
            fig=plt.gcf(),
            title=title,
            fig_name=metrics_list[j],
            fig_format="png",
            fig_dpi=600,
            fig_size=[10, 10],
        )
    
    # Plot the joint amplitudes
    for idx, label in enumerate(["w^{ipsi}", "w^{contra}"]):
        plt.figure(f"Joint Amplitudes {label}", figsize=(10, 10))
        legend_entries = []
        for i in range(15):
            # Plot for w^ipsi or w^contra
            plt.plot(
                mat_weights[n_weights*idx:n_weights*(1+idx), idx],
                mat_joint_amplitudes[n_weights*idx:n_weights*(1+idx), i],
                marker='o',
                markersize=5,
            )
            legend_entries.append(f"Joint {i+1}")
            
            plt.legend(
                legend_entries,
                loc='center left',
                bbox_to_anchor=(1.0, 0.5),
                fontsize='small',
            )
            plt.xlabel(f"Weight (${label}$)")
            plt.ylabel("Joint Amplitude")
            plt.minorticks_on()
            plt.grid(which='major', color='darkgrey', linestyle='-', linewidth=0.75)
            plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)

        save_figures(
            logdir=logdir,
            fig=plt.gcf(),
            title=title,
            fig_name=f"joint_amplitudes_{label}",
            fig_format="png",
            fig_dpi=600,
            fig_size=[10, 10],
        )


def exercise6(run_sim = True, run_plot = True):

    pylog.info("Ex 6")
    log_path = './logs/exercise6/'  # Path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    # Define the parameters for the simulation
    drive = 10
    timestep = 0.001
    n_iterations = 100001
    cpg_amplitude_gain = np.array(
            [
                0.00824, 0.00328, 0.00328, 0.00370, 0.00451,
                0.00534, 0.00628, 0.00680, 0.00803, 0.01084,
                0.01115, 0.01149, 0.01655,
            ])
    n_weights = 51; # number of weights to be simulated

    if run_sim:
        pars_list = [
            SimulationParameters(
                simulation_i = i,
                controller="abstract oscillator",
                n_iterations=n_iterations,
                timestep=timestep,
                compute_metrics="all",
                headless=True,
                video_record=False,
                log_path=log_path,
                return_network=False,
                print_metrics=False,
                feedback_weights_ipsi = weight,
                feedback_weights_contra = 0,
                ws_ref = ws_ref,
                drive = drive,
                cpg_amplitude_gain = cpg_amplitude_gain,
            )
            for i, weight in enumerate(np.linspace(-1, 1, n_weights))      
        ]

        # Run the simulation
        controllers = run_multiple(pars_list, num_process=num_process)


        pars_list = [
            SimulationParameters(
                simulation_i = n_weights+i,
                controller="abstract oscillator",
                n_iterations=n_iterations,
                timestep=timestep,
                compute_metrics="all",
                headless=True,
                video_record=False,
                log_path=log_path,
                return_network=False,
                print_metrics=False,
                feedback_weights_ipsi = 0,
                feedback_weights_contra = weight,
                ws_ref = ws_ref,
                drive = drive,
                cpg_amplitude_gain = cpg_amplitude_gain,

            )
            for i, weight in enumerate(np.linspace(-1, 1, n_weights))      
        ]

        # Run the simulation
        controllers = run_multiple(pars_list, num_process=num_process)

    # Plot the results
    if run_plot:
        plot_E6(n_weights, log_path, "exercise6")

if __name__ == '__main__':
    run_sim = False
    run_plot = True

    exercise6(run_sim,run_plot)
