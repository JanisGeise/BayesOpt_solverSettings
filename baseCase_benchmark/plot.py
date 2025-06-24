import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from yaml import safe_load
import sys
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
try:
    with open(config_file, "r") as cf:
        config = safe_load(cf)
except Exception as e:
    print(e)

pwd = os.getcwd()
fol = os.path.join(pwd, "benchmark_time_info")
folder = os.listdir(fol)
data_li = []
t_plot_li = []
t_cum_plot_li = []
t_cum_plot_box_li = []
t_cum_plot_li_max = []
t_cum_plot_li_min = []
print(folder)
for fold in folder:
    run = fold
    base_sim_path = os.path.join(fol, run)
    base_timing = pd.read_csv(
        os.path.join(base_sim_path),
        header=None,
        sep=r"\s+",
        skiprows=1,
        usecols=[0, 1],
        names=["t", "t_cpu_cum"],
    )
    data_li.append(base_timing)

sim_dur = float(config["simulation"]["duration"])
opt_dur = float(config["optimization"]["duration"])
dt = float(config["optimization"]["deltaT"])

steps = [30, 100, 200, 300, 500]
# normalizer = [sim_dur - 0.5 * step * dt * 0.9 for step in steps]

for step in steps:
    mean_li = []
    for base_timing in data_li:
        f = interp1d(base_timing.t.values, base_timing.t_cpu_cum.values)
        t_inter = np.linspace(dt, sim_dur, int(sim_dur / dt))
        t_cpu_cum_inter = f(t_inter)

        # t_plot = base_timing.t.values[step - 1 :: step] - step * 0.001
        # t_cum_plot = base_timing.t_cpu_cum.values[step - 1 :: step]

        t_plot = t_inter[step - 1 :: step] - step * dt
        t_cum_plot = t_cpu_cum_inter[step - 1 :: step]

        t_cum_plot = (
            np.concatenate((t_cum_plot[:1], t_cum_plot[1:] - t_cum_plot[:-1])) / step
        )
        mean_li.append(t_cum_plot)

    box_li = np.array(mean_li)
    mean_array = np.mean(np.stack(mean_li), axis=0)
    min_array = mean_array - np.min(np.stack(mean_li), axis=0)
    max_array = np.max(np.stack(mean_li), axis=0) - mean_array

    t_cum_plot_li.append(mean_array)
    t_cum_plot_box_li.append(box_li)
    t_cum_plot_li_max.append(max_array)
    t_cum_plot_li_min.append(min_array)
    t_plot_li.append(t_plot)

for ind in range(len(t_cum_plot_li)):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    x_centers = t_plot_li[ind] + steps[ind] * dt * 0.9 / 2
    normalizer = x_centers[-1] + steps[ind] * dt * 0.9 / 2
    x_centers = x_centers / normalizer

    flierprops = dict(
        marker="o",
        markersize=1,
        markerfacecolor="red",
        linestyle="none",
        markeredgecolor="red",
    )
    boxprops = dict(linewidth=0.5, color="black")
    ax.boxplot(
        t_cum_plot_box_li[ind],
        positions=x_centers,
        widths=steps[ind] * dt * 0.9 / normalizer,
        patch_artist=False,
        flierprops=flierprops,
        boxprops=boxprops,
    )
    ax.bar(
        t_plot_li[ind] / normalizer,
        t_cum_plot_li[ind],
        width=steps[ind] * dt * 0.9 / normalizer,
        align="edge",
    )

    ax.set_xlabel(r"$\tilde{t}$")
    step = steps[ind]
    ax.set_ylabel(r"$T_{{{stept}\Delta t}}$".format(stept=step))
    ax.set_title("Execution time for buffet base case - {} time steps".format(step))
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig.savefig(
        "exe_time_{}_time_steps.svg".format(steps[ind]),
        bbox_inches="tight",
        transparent=True,
    )

fig1, ax1 = plt.subplots(figsize=(10, 5))
colors = ["red", "blue", "black", "purple", "green"]
for ind in range(len(t_cum_plot_li)):
    ax1.plot(
        t_plot_li[ind],
        t_cum_plot_li[ind],
        marker=".",
        label="{}".format(steps[ind]),
        color=colors[ind],
    )


ax1.set_xlabel(r"$\tilde{t}$")
ax1.set_ylabel(r"$T_{\Delta t}$")
ax1.set_title("Execution time for base case - different time steps")
ax1.legend()
fig1.savefig("exe_time_all_timesteps.svg", bbox_inches="tight", transparent=True)
