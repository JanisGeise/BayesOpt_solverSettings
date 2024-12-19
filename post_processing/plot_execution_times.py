"""
    plot the execution times of the simulation per timestep
    TODO: clean up etc., avg. over multiple runs, plot total exec tie with errorbars as in master thesis
"""
import matplotlib.pyplot as plt

from os import makedirs
from pandas import read_csv
from os.path import join, exists

if __name__ == "__main__":
    mesh = 1
    load_path = join("..", "run")
    save_path = join("..", "run", "plots")
    cases = [f"cylinder_2D_Re100_default_mesh{mesh}", f"cylinder_2D_Re100_GaussSeidel_mesh{mesh}",
             f"cylinder_2D_Re100_FDIC_mesh{mesh}"]
    legend = ["DICGaussSeidel (default)", "GaussSeidel", "FDIC"]

    # create save directory if not exists
    if not exists(save_path):
        makedirs(save_path)

    # use latex fonts
    plt.rcParams.update({"text.usetex": True})

    # load the time info function object TODO: extend for optimized case with multiple separate timeInfo.dat files
    exec_times = [read_csv(join(load_path, c, "postProcessing", "time", "0", "timeInfo.dat"), header=None, sep=r"\s+",
                           skiprows=1, usecols=[0, 1, 3], names=["t", "t_cpu_cum", "t_cpu"]) for c in cases]

    # plot the results
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for e in exec_times:
        ax.plot(e.t, e.t_cpu)
    ax.set_xlim(exec_times[0].t.min(), exec_times[0].t.max())
    ax.set_xlabel(r"$t$ $[s]$")
    ax.set_ylabel(r"$t / \Delta t$ $[s]$")
    ax.legend(legend, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.subplots_adjust()
    plt.savefig(join(save_path, f"execution_time_per_time_step_mesh{mesh}_comparison_smoother.png"), dpi=340)
    plt.close("all")
