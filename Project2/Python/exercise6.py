import os
import numpy as np
import matplotlib.pyplot as plt
from farms_core import pylog

from simulation_parameters import SimulationParameters
from util.run_closed_loop import run_multiple


# -----------------------------------------------------------------------------
# Exercise 6 – Separate ipsi / contra gain sweeps + aesthetic plots
# -----------------------------------------------------------------------------

def exercise6():
    pylog.info("Exercise 6: starting ipsilateral and contralateral gain sweeps …")

    # ------------------------------------------------------------------
    # Paths & config
    # ------------------------------------------------------------------
    log_path   = "./logs/exercise6/"
    plots_path = os.path.join(log_path, "plots")  # <‑‑ new sub‑folder
    os.makedirs(plots_path, exist_ok=True)

    gains = np.linspace(-1.0, 1.0, 51)  # −1.00, −0.80, …, +1.00 (step 0.2)

    pars_list: list[SimulationParameters] = []
    gain_tags: list[tuple[str, float]] = []  # (mode, gain)

    # ------------------------------------------------------------------
    # 1) Ipsilateral sweep (contralateral gain = 0)
    # ------------------------------------------------------------------
    for i, g in enumerate(gains):
        pars_list.append(
            SimulationParameters(
                controller="abstract oscillator",
                simulation_i=i,
                simulation_name=f"controller{i}",
                log_path=log_path,
                return_network=True,
                compute_metrics="all",
                feedback_weights_ipsi=g,
                feedback_weights_contra=0.0,
            )
        )
        gain_tags.append(("ipsi", g))

    offset = len(gains)

    # ------------------------------------------------------------------
    # 2) Contralateral sweep (ipsilateral gain = 0)
    # ------------------------------------------------------------------
    for j, g in enumerate(gains):
        idx = offset + j
        pars_list.append(
            SimulationParameters(
                controller="abstract oscillator",
                simulation_i=idx,
                simulation_name=f"controller{idx}",
                log_path=log_path,
                return_network=True,
                compute_metrics="all",
                feedback_weights_ipsi=0.0,
                feedback_weights_contra=g,
            )
        )
        gain_tags.append(("contra", g))

    # ------------------------------------------------------------------
    # Run all simulations in parallel
    # ------------------------------------------------------------------
    networks = run_multiple(pars_list, num_process=min(6, len(pars_list)))

    # ------------------------------------------------------------------
    # Collect scalar metrics, split by sweep type
    # ------------------------------------------------------------------
    data_ipsi, data_contra = {"gains": []}, {"gains": []}

    for (mode, g), net in zip(gain_tags, networks):
        if not hasattr(net, "metrics"):
            continue
        target = data_ipsi if mode == "ipsi" else data_contra
        target["gains"].append(g)
        for m, val in net.metrics.items():
            if np.isscalar(val):
                target.setdefault(m, []).append(val)

    # sort gains & corresponding metric arrays
    for data in (data_ipsi, data_contra):
        order = np.argsort(data["gains"])
        data["gains"] = np.asarray(data["gains"])[order]
        for k in list(data.keys()):
            if k == "gains":
                continue
            v = np.asarray(data[k])
            if len(v) == len(order):
                data[k] = v[order]

    # ------------------------------------------------------------------
    # Aesthetic plotting helper – saves into plots_path
    # ------------------------------------------------------------------
    def _pretty_plot(x, y, xlabel, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, marker="o", linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(-1.0, 1.01, 0.25))
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_path, filename), dpi=600)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Generate plots for each scalar metric
    # ------------------------------------------------------------------
    for metric in set(data_ipsi) | set(data_contra):
        if metric == "gains":
            continue
        if metric in data_ipsi:
            _pretty_plot(
                data_ipsi["gains"],
                data_ipsi[metric],
                "Ipsilateral feedback gain",
                metric.replace("_", " ").title(),
                f"{metric.replace('_', ' ').title()} vs Ipsilateral gain",
                f"ipsi_{metric}.png",
            )
        if metric in data_contra:
            _pretty_plot(
                data_contra["gains"],
                data_contra[metric],
                "Contralateral feedback gain",
                metric.replace("_", " ").title(),
                f"{metric.replace('_', ' ').title()} vs Contralateral gain",
                f"contra_{metric}.png",
            )

    pylog.success("Exercise 6 complete – plots saved in ./logs/exercise6/plots/")


if __name__ == "__main__":
    exercise6()

