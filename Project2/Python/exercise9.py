
from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog
import numpy as np
import matplotlib.pyplot as plt
from plotting_common import plot_time_histories
from util.rw import load_object

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]
weight = 1
joint_poses = 0.1*np.ones(15)
n_iterations = 500001
n_timestep = 0.0001

def plot_Exercise9(controller, pars, log_path):
    # Plot oscillator phases
    plt.figure(figsize=(15, 8))
    for i in range(min(2, controller.oscillator_phase_all.size)):
        phase = controller.state[:, controller.oscillator_phase_all[i]]# % (2 * np.pi)
        plt.plot(controller.times, phase, label=f'Oscillator {i}')
    plt.title('Oscillator Phases')
    plt.xlabel('Time [s]')
    plt.ylabel('Phase [rad]')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0,4)
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'oscillator_phases.png'), dpi=600, bbox_inches='tight')
    plt.close()
    
    # Plot oscillator amplitudes
    plt.figure(figsize=(15, 8))
    for i in range(min(6, controller.oscillator_amplitude_all.size)):
        amp = controller.state[:, controller.oscillator_amplitude_all[i]]
        plt.plot(controller.times, amp, label=f'Oscillator {i}')
    plt.title('Oscillator Amplitudes')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'oscillator_amplitudes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot motor outputs
    plt.figure(figsize=(15, 8))
    for i in range(min(4, controller.motor_l.size)):
        plt.plot(controller.times, controller.motor_out[:, controller.motor_l[i]], '--', label=f'L{i}')
        plt.plot(controller.times, controller.motor_out[:, controller.motor_r[i]], '-', label=f'R{i}')
    plt.title('Motor Outputs')
    plt.xlabel('Time [s]')
    plt.ylabel('Activation')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'motor_outputs.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot motor output differences
    plt.figure(figsize=(15, 8))
    motor_diff = controller.motor_out[:, controller.motor_l[:6]] - controller.motor_out[:, controller.motor_r[:6]]
    for i in range(motor_diff.shape[1]):
        plt.plot(controller.times, motor_diff[:, i], label=f'Joint {i}')
    plt.title('Motor Output Differences (Left - Right)')
    plt.xlabel('Time [s]')
    plt.ylabel('Activation Difference')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'motor_differences.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot joint angles
    plt.figure(figsize=(15, 8))
    active = controller.joints_positions[:, :pars.n_joints]
    time_vec = controller.times[:active.shape[0]]
    for j in range(min(4, active.shape[1])):
        plt.plot(time_vec, active[:, j], label=f'Joint {j}')
    plt.title('Joint Angles')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'joint_angles.png'), dpi=300, bbox_inches='tight')
    plt.close()


def exercise9():

    pylog.info("Ex 9")
    log_path = './logs/exercise9/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    
    pars1 = SimulationParameters(
        n_iterations=n_iterations,             
        timestep=n_timestep,
        controller='abstract oscillator',
        log_path=log_path,
        simulation_i=0,
        compute_metrics='all',
        print_metrics=True,
        return_network=True,
        feedback_weights_ipsi=0,
        feedback_weights_contra=-weight,
        amplitude_rates = 0,
        drive = 0,
        ws_ref=ws_ref,
        joint_poses = joint_poses,
        weights_body2body = 0,
        weights_body2body_contralateral = 0,
    )

    controller1 = run_single(pars1)
    
    # pars = SimulationParameters(
    #     n_iterations=n_iterations,             
    #     timestep=n_timestep,
    #     controller='abstract oscillator',
    #     log_path=log_path,
    #     simulation_i=1,
    #     compute_metrics='all',
    #     print_metrics=True,
    #     return_network=True,
    #     feedback_weights_ipsi=weight,
    #     feedback_weights_contra=0,
    #     amplitude_rates = 0,
    #     drive = 0,
    #     ws_ref=ws_ref,
    #     joint_poses = joint_poses,
    # )

    # controller2 = run_single(pars)
    
    # pars = SimulationParameters(
    #     n_iterations=n_iterations,             
    #     timestep=n_timestep,
    #     controller='abstract oscillator',
    #     log_path=log_path,
    #     simulation_i=2,
    #     compute_metrics='all',
    #     print_metrics=True,
    #     return_network=True,
    #     feedback_weights_ipsi=0,
    #     feedback_weights_contra=0,
    #     amplitude_rates = 0,
    #     drive = 0,
    #     ws_ref=ws_ref,
    #     joint_poses = joint_poses,
    # )

    # controller3 = run_single(pars)

    for i in range(1):
        controller = eval(f'controller{i+1}')
        pars = eval(f'pars{i+1}')
        plot_Exercise9(controller, pars, log_path)

if __name__ == '__main__':

    exercise9()

