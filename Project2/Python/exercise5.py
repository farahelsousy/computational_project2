import numpy as np
import matplotlib.pyplot as plt
import os
from farms_core import pylog  # fixed import
from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_single  # use closed-loop runner
from util.zebrafish_hyperparameters import define_hyperparameters

# Define general parameters for the simulation
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]

def exercise5():
    pylog.info("Ex 5: Closed-loop CPG test")
    
    # Set up logging
    log_path = './logs/exercise5/'
    os.makedirs(log_path, exist_ok=True)

    # Create simulation parameters with default values
    pars = SimulationParameters(
        n_iterations=10001,             # 10 s @ 0.001 s timestep
        timestep=0.001,
        n_joints=13,
        controller='abstract oscillator',
        log_path=log_path,
        simulation_i=0,
        compute_metrics='all',
        print_metrics=True,
        return_network=True,
        headless=False,
        video_record=True,
        video_name='exercise5_cpg_swim',
        video_fps=50,
        phase_lag_body=2*np.pi, 
        feedback_weights_ipsi=0.25,
        feedback_weights_contra=-0.25,
        ws_ref=ws_ref
    )

    try:
        # Run the closed-loop CPG simulation
        controller = run_single(pars)

        if controller and hasattr(controller, 'metrics'):
            pylog.info("Simulation finished successfully")
            
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
            if hasattr(controller, 'joints_positions'):
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
            else:
                pylog.warning("'joints_positions' not found in controller.")
            
            pylog.info("Analysis complete. Results saved in {}".format(log_path))

        else:
            pylog.error("Failed to retrieve valid controller object from simulation.")
    except Exception as e:
        pylog.error(f"Simulation failed! Error: {e}", exc_info=True)

if __name__ == '__main__':
    exercise5()