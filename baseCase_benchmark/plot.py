import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

pwd = os.getcwd()
fol = os.path.join(pwd, "openfoam_cases")
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
        os.path.join(base_sim_path, "postProcessing", "time", "0", "timeInfo.dat"),
        header=None,
        sep=r"\s+",
        skiprows=1,
        usecols=[0, 1],
        names=["t", "t_cpu_cum"],
        )
    data_li.append(base_timing)
steps = [25, 50, 75, 100]
for step in steps:
    mean_li = []
    for base_timing in data_li:
        t_plot = base_timing.t.values[step-1::step] - step*0.001
        t_cum_plot = base_timing.t_cpu_cum.values[step-1::step]
        t_cum_plot = np.concatenate((t_cum_plot[:1], t_cum_plot[1:]-t_cum_plot[:-1]))/step
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
    x_centers = t_plot_li[ind]+steps[ind]*0.001*0.9/2
    flierprops = dict(marker='o', markersize=1, markerfacecolor='red', linestyle='none', markeredgecolor="red")
    boxprops = dict(linewidth=0.5, color='black')
    ax.boxplot(t_cum_plot_box_li[ind], positions=x_centers, widths=steps[ind]*0.001*0.9, patch_artist=False, flierprops=flierprops, boxprops=boxprops)
    ax.bar(t_plot_li[ind], t_cum_plot_li[ind], width=steps[ind]*0.001*0.9, align="edge")
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel(r"$\tilde{t}$")
    step = steps[ind]
    ax.set_ylabel(r"$T_{{{stept}\Delta t}}$".format(stept = step))
    ax.set_title("Execution time for base case - {} time steps".format(step))
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    
    ax.set_xticklabels([1, 2, 3, 4, 5, 6])
    fig.savefig("exe_time_{}_time_steps.svg".format(steps[ind]), bbox_inches="tight", transparent=True)

fig1, ax1 = plt.subplots( figsize=(10, 5))
colors = ["red", "blue", "black", "purple"]
for ind in range(len(t_cum_plot_li)):
    ax1.plot(t_plot_li[ind], t_cum_plot_li[ind], marker=".", label = "{}".format(steps[ind]), color = colors[ind])


ax1.set_xlabel(r"$\tilde{t}$")
ax1.set_ylabel(r"$T_{\Delta t}$")
ax1.set_title("Execution time for base case - different time steps")
ax1.legend()
fig1.savefig("exe_time_all_timesteps.svg", bbox_inches="tight", transparent=True)
