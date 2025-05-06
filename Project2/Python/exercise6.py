import os
import numpy as np
import matplotlib.pyplot as plt
from farms_core import pylog
from util.run_closed_loop import run_multiple
from simulation_parameters import SimulationParameters
from util.zebrafish_hyperparameters import define_hyperparameters

hyper = define_hyperparameters()
FEEDBACK_GAIN_REF = 1.0 / np.mean(hyper["REF_JOINT_AMP"])
WEIGHTS = np.linspace(-1, 1, 9)
N_JOINTS = 13
DEFAULT_NOM_AMP = np.ones(2 * N_JOINTS)
DEFAULT_A_RATE = 1.0

NEUR = ("neur_frequency", "neur_amp", "neur_twl")
MECH = ("mech_mean_frequency", "mech_mean_amplitude", "mech_speed_fwd",
        "mech_speed_lat", "mech_cot", "mech_energy", "mech_torque", "mech_twl")
LABEL = {
    'neur_frequency': 'Neural Frequency [Hz]',
    'neur_amp': 'Mean Neural Amplitude',
    'neur_twl': 'Neural Total Wave Lag',
    'mech_mean_frequency': 'Mean Mechanical Frequency [Hz]',
    'mech_mean_amplitude': 'Mean Mechanical Amplitude [rad]',
    'mech_speed_fwd': 'Forward Speed',
    'mech_speed_lat': 'Lateral Speed',
    'mech_cot': 'Cost of Transport',
    'mech_energy': 'Energy Consumption',
    'mech_torque': 'Sum of Torques',
    'mech_twl': 'Mechanical Total Wave Lag',
}

def build_par(w, side, i, log_root):
    """Build simulation parameters for a specific weight and side."""
    w_ipsi, w_contra = (w, 0) if side == "ipsi" else (0, w)
    
    # Create a more descriptive name for the controller
    # Format the weight to 2 decimal places without trailing zeros
    weight_str = f"{w:.2f}".rstrip('0').rstrip('.') if '.' in f"{w:.2f}" else f"{w:.2f}"
    controller_name = f"{side}_w{weight_str}"
    
    return SimulationParameters(
        n_iterations=10001,
        timestep=0.001,
        n_joints=N_JOINTS,
        controller="abstract oscillator",
        simulation_i=i,
        log_path=log_root + "/" if not log_root.endswith("/") else log_root,
        return_network=True,
        headless=True,
        video_record=False,
        feedback_weights_ipsi=w_ipsi,
        feedback_weights_contra=w_contra,
        feedback_gain_ref=FEEDBACK_GAIN_REF,
        nominal_amplitude=DEFAULT_NOM_AMP,
        amplitude_rates=DEFAULT_A_RATE,
        simulation_name=controller_name  # Add descriptive name
    )

def run_sweep(side, log_root):
    """Run a sweep of simulations for a specific side (ipsi/contra)."""
    pylog.info(f"Starting {side} sweep")
    
    # Create a dedicated folder for this sweep
    sweep_folder = os.path.join(log_root, f"{side}_sweep")
    os.makedirs(sweep_folder, exist_ok=True)
    
    # Create parameters for all weights with the correct folder
    params = []
    for i, w in enumerate(WEIGHTS):
        # Create a dedicated subfolder for each controller to ensure proper organization
        param = build_par(w, side, i, sweep_folder)
        params.append(param)
    
    # Run simulations with fewer parallel processes
    controllers = run_multiple(params, num_process=6)
    
    # Initialize arrays to store metrics
    metrics = {k: np.zeros(len(WEIGHTS)) for k in NEUR + MECH}
    
    # Process results
    for i, (w, ctrl) in enumerate(zip(WEIGHTS, controllers)):
        if ctrl is None:
            pylog.warning(f"Controller for {side} sweep, weight {w:.2f} failed")
            continue
            
        # Store metrics in arrays
        for k in NEUR + MECH:
            metrics[k][i] = ctrl.metrics.get(k, np.nan)
    
    return metrics

def plot(metric, ipsi, contra, root):
    """Plot comparison between ipsi and contra sweeps."""
    try:
        # Create figure with larger size
        plt.figure(figsize=(12, 8))
        
        # Plot data with different styles and clearer labels
        plt.plot(WEIGHTS, ipsi, 'o-', label='Ipsilateral feedback', linewidth=2, markersize=8, color='blue')
        plt.plot(WEIGHTS, contra, 's--', label='Contralateral feedback', linewidth=2, markersize=8, color='red')
        
        # Add labels and title
        plt.xlabel('Raw feedback weight (×k_ref)', fontsize=14)
        plt.ylabel(LABEL.get(metric, metric), fontsize=14)
        plt.title(f'{LABEL.get(metric, metric)} vs Feedback Weight', fontsize=16)
        
        # Add grid and legend
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Ensure directory exists for plots
        plots_dir = os.path.join(root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save the plot with a more descriptive filename
        plot_path = os.path.join(plots_dir, f'{metric}_vs_weight.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        pylog.info(f"Saved plot to {plot_path}")
        
        # Show the plot
        plt.show()
        
        # Close the figure
        plt.close()
        
    except Exception as e:
        pylog.error(f"Error plotting {metric}: {str(e)}")

def exercise6():
    """Main function to run the exercise."""
    log_root = "./logs/exercise6/"
    os.makedirs(log_root, exist_ok=True)
    
    # Run sweeps
    pylog.info("Running ipsilateral feedback sweep")
    ipsi_metrics = run_sweep("ipsi", log_root)
    
    pylog.info("Running contralateral feedback sweep")
    contra_metrics = run_sweep("contra", log_root)
    
    # Save metrics data for future reference
    np.savez(os.path.join(log_root, "sweep_metrics.npz"), 
             weights=WEIGHTS,
             ipsi_metrics=ipsi_metrics,
             contra_metrics=contra_metrics)
    
    # Generate plots
    for k in NEUR + MECH:
        pylog.info(f"Generating plot for {k}")
        plot(k, ipsi_metrics[k], contra_metrics[k], log_root)
    
    pylog.info(f"Analysis complete. Results saved in {log_root}")

if __name__ == "__main__":
    exercise6()
