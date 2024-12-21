"""
    plot the execution times of the simulation per timestep and compare the overall execution times
"""
import matplotlib.pyplot as plt

from glob import glob
from os import makedirs
from torch import tensor
from os.path import join, exists
from scipy.ndimage import gaussian_filter1d
from pandas import read_csv, concat, DataFrame


def load_execution_times(load_dir: str) -> DataFrame:
    dirs = sorted(glob(join(load_dir, "postProcessing", "time", "*")), key=lambda x: float(x.split("/")[-1]))
    res = [read_csv(join(d, "timeInfo.dat"), header=None, sep=r"\s+", skiprows=1, usecols=[0, 3], names=["t", "t_cpu"])
           for d in dirs]
    return res[0] if len(res) == 1 else concat(res)


def get_cumulative_execution_time(load_dir: str) -> float:
    dirs = sorted(glob(join(load_dir, "postProcessing", "time", "*")), key=lambda x: float(x.split("/")[-1]))
    res = [read_csv(join(d, "timeInfo.dat"), header=None, sep=r"\s+", skiprows=1, usecols=[1],
                    names=["t_cpu_cum"]) for d in dirs]

    return res[0]["t_cpu_cum"].iloc[-1] if len(res) == 1 else sum([r["t_cpu_cum"].iloc[-1] for r in res])


if __name__ == "__main__":
    mesh = 0
    load_path = join("..", "run")
    save_path = join("..", "run", f"plots_first_tests_mesh{mesh}")
    cases = [f"default_mesh{mesh}", f"first_tests_mesh{mesh}"]
    legend = ["default", "first test"]

    # end times of the chosen intervals for which the optimization was executed, if nothing was changed use empty list
    interval = [[], [1, 3]]

    # create save directory if not exists
    if not exists(save_path):
        makedirs(save_path)

    # use latex fonts and default color cycle
    plt.rcParams.update({"text.usetex": True})
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

    # get the cumulative execution time for all cases, all runs for each case and all intervals for each run
    t_cum = []
    for c in cases:
        t_cum.append(tensor([get_cumulative_execution_time(run) for run in glob(join(load_path, c, "run*"))]))

    # plot the mean cumulative execution time and corresponding standard deviation, scaled wrt default
    default_idx = 0
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for i, t in enumerate(t_cum):
        ax.errorbar(legend[i], t.mean() / t_cum[default_idx].mean(), yerr=t.std() / t.mean(), barsabove=True, fmt="o",
                    capsize=5)

    ax.set_ylabel("$t / t_{base}$")
    fig.tight_layout()
    ax.grid(visible=True, which="major", linestyle="-", alpha=0.45, color="black", axis="y")
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(visible=True, which="minor", linestyle="--", alpha=0.35, color="black", axis="y")
    plt.savefig(join(save_path, f"comparison_overall_execution_times.png"), dpi=340)
    plt.close("all")

    # compare the execution time per time step for a single run
    run = 1
    exec_times = [load_execution_times(join(load_path, c, f"run{run}")) for c in cases]

    # plot the results
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for i, e in enumerate(exec_times):
        # smooth the results for better comparison
        ax.plot(e.t, gaussian_filter1d(e.t_cpu.values, 3), zorder=10, color=color[i], label=legend[i])
        ax.plot(e.t, e.t_cpu.values, alpha=0.2, color=color[i])
        if interval[i]:
            [ax.axvline(iv, color="red", ls=":") for iv in interval[i]]
    ax.set_xlim(exec_times[0].t.min(), exec_times[0].t.max())
    ax.set_xlabel(r"$t$ $[s]$")
    ax.set_ylabel(r"$t$ per $\Delta t$ $[s]$")
    ax.legend(ncol=4, loc="upper right")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_path, f"execution_time_per_time_step.png"), dpi=340)
    plt.close("all")
