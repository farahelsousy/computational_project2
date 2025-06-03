"""
exercise9.py  –  Part 7 sweep with random initial phases
-------------------------------------------------------
Runs 3 × 10 closed-loop simulations:

    1.  b2b only               (w_b2b = 30, w_contra = 0)
    2.  contralateral only     (w_b2b = 0,  w_contra = 10)
    3.  no CPG couplings       (w_b2b = 0,  w_contra = 0)

Each simulation gets a fresh random set of initial phases in [0, 2π].
After completion, mean ± SD of every scalar metric are written to
logs/exercise9/summary_metrics.json
"""

import os
import json
import numpy as np
from collections import defaultdict
from farms_core import pylog

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple            # <–– parallel runner
from util.zebrafish_hyperparameters import define_hyperparameters

# --------------------------------------------------------------------------- #
ws_ref = define_hyperparameters()["ws_ref"]  # reference stretch weight

LOG_ROOT = "./logs/exercise9"                # all outputs rooted here
os.makedirs(LOG_ROOT, exist_ok=True)


def build_parameter_list() -> list[SimulationParameters]:
    """Create SimulationParameters for 3 cases × 10 seeds."""
    cases = {
        "b2b_only":      dict(weights_body2body=30, weights_body2body_contralateral=0),
        "contra_only":   dict(weights_body2body=0,  weights_body2body_contralateral=10),
        "no_cpg":        dict(weights_body2body=0,  weights_body2body_contralateral=0),
    }

    all_pars = []
    sim_id = 0
    for case_name, w in cases.items():
        case_log = os.path.join(LOG_ROOT, case_name)
        os.makedirs(case_log, exist_ok=True)

        for seed in range(1):
            pars = SimulationParameters(
                # --- basic sim setup -------------------------------------------------
                n_iterations=20001,           # 50 s @ 0.001 s
                timestep=0.001,
                n_joints=13,
                controller="abstract oscillator",
                # --- logging ---------------------------------------------------------
                log_path=case_log,
                simulation_i=seed,
                drive=10,
                compute_metrics="all",
                print_metrics=False,
                return_network=True,
                headless = False,
                # --- video / plotting off for batches --------------------------------
                video_record=True,
                # --- feedback + CPG weights -----------------------------------------
                feedback_weights_ipsi=0.25,
                feedback_weights_contra=-0.25,
                ws_ref=ws_ref,
                phase_lag_body=2 * np.pi,
                **w,                               # inject coupling weights
                # --- random initial phases ------------------------------------------
                initial_phases=np.random.rand(26) * 2 * np.pi,
            )
            all_pars.append(pars)
            sim_id += 1

    return all_pars


def summarise_metrics(par_list, controller_list):
    """Compute mean ± SD of every scalar metric per coupling case."""
    grouped = defaultdict(list)

    for pars, ctrl in zip(par_list, controller_list):
        if ctrl is None or not hasattr(ctrl, "metrics"):
            pylog.warning(f"Simulation {pars.simulation_i} in {pars.log_path} "
                          "returned no metrics; skipping.")
            continue
        case = os.path.basename(os.path.normpath(pars.log_path))
        grouped[case].append(ctrl.metrics)

    summary = {}
    for case, metr_list in grouped.items():
        keys = metr_list[0].keys()
        summary[case] = {}
        for k in keys:
            vals = np.array([m[k] for m in metr_list], dtype=float)
            if vals.ndim == 1:                      # scalar metrics
                summary[case][f"{k}_mean"] = float(vals.mean())
                summary[case][f"{k}_std"]  = float(vals.std(ddof=1))
            # ignore vector metrics (joint-by-joint) for table clarity

    out_file = os.path.join(LOG_ROOT, "summary_metrics.json")
    with open(out_file, "w") as fp:
        json.dump(summary, fp, indent=4)
    pylog.info(f"Mean ± SD metrics written to {out_file}")

    # Pretty print to console
    for case, d in summary.items():
        pylog.info(f"\n--- {case} ---")
        for k, v in d.items():
            pylog.info(f"{k:25s}: {v:.4g}")


def exercise9():
    pylog.info("Exercise 9 sweep: 3 coupling cases × 10 random seeds")

    param_list = build_parameter_list()
    pylog.info(f"Launching {len(param_list)} simulations …")

    controllers = run_multiple(param_list,num_process=1) 
    summarise_metrics(param_list, controllers)

    pylog.info("Exercise 9 sweep complete.")


if __name__ == "__main__":
    exercise9()
