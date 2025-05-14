import sys
from os.path import join
from yaml import safe_load
from ax.service.ax_client import AxClient
from ax.modelbridge.registry import Models
from ax.plot.contour import interact_contour_plotly, interact_contour
from ax.utils.report.render import render_report_elements
from ax.plot.render import plot_config_to_html
from ax.plot.render import plotly_offline
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from os import makedirs, getcwd
import pandas as pd
from ax.core.observation import ObservationFeatures
from ax.modelbridge.registry import Generators
plt.style.use('dark_background')
plt.rcParams.update({
    "text.usetex": False
})

def plot_trial_vs_base(config, ax_clients):
    base_sim_path = join(f"{config['experiment']['exp_path']}", "base_sim")
    base_timing = read_csv(
        join(base_sim_path, "postProcessing", "time", "0", "timeInfo.dat"),
        header=None,
        sep=r"\s+",
        skiprows=1,
        usecols=[0, 1],
        names=["t", "t_cpu_cum"],
    )
    dur = float(config['optimization']['duration'])
    step = int(dur/float(config['optimization']['deltaT']))
    t_plot = base_timing.t.values[step-1::step] - dur
    t_cum_plot = base_timing.t_cpu_cum.values[step-1::step]
    t_cum_plot = np.concatenate((t_cum_plot[:1], t_cum_plot[1:]-t_cum_plot[:-1]))/step
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.bar(t_plot, t_cum_plot, width=dur*0.9, align="center")
    ax.set_xlim(0, 6)
    ax.set_xlabel(r"$\tilde{t}$")
    ax.set_ylabel(r"$T_{{{ste}\Delta t}}$".format(ste = step))


    for i, st_i in enumerate(config["optimization"]["startTime"][:]):

        data = ax_clients[i].experiment.fetch_data().df["mean"].values
        if i==0:
            ax.scatter([float(st_i)]*len(data), data, marker="x", s=10, c="C3", label="trials", linewidth=0.5, alpha=0.5)
        else:
            ax.scatter([float(st_i)]*len(data), data, marker="x", s=10, c="C3", linewidth=0.5, alpha=0.5)
    ax.legend()
    path = join(getcwd(), "output", "trial_vs_base")
    makedirs(path, exist_ok=True)
    fig.savefig(join(path, "execution_time_dt_opt.svg"), bbox_inches="tight", transparent=True)

def plot_best_params(config, ax_clients):
    dic_best = {}
    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = ax_clients[i]
        dic_best[float(st_i)] = ax_client.get_best_parameters()


    parallel_data = []
    for x in dic_best.keys():
        parallel_dic = dic_best[x][0]
        parallel_dic["interval"] = x
        parallel_data.append(parallel_dic)
        
    df = pd.DataFrame(parallel_data)

    li = list(dic_best[float(st_i)][0].keys())
    li.remove("interval")
    plot_li = ['interval']
    plot_li.extend(li)
    plot_df = df[plot_li]

    path = join(getcwd(), "output", "best_params")
    makedirs(path, exist_ok=True)

    for xi in li:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(plot_df["interval"], plot_df[xi], marker="o")
        ax1.set_title("Best parameter - {} - for different intervals".format(xi))
        ax1.set_xlabel("interval")
        ax1.set_ylabel(xi)
        fig1.savefig(join(path, "parallel_plot_best_params_{}.svg".format(xi)), bbox_inches="tight", transparent=True)

def plot_trialvsobj(config, ax_clients):
    data_li = []

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = ax_clients[i]
        df = ax_client.experiment.fetch_data().df
        df["best"] = df["mean"].cummin()
        data_li.append(df)

    path = join(getcwd(), "output", "trialvsobj")
    makedirs(path, exist_ok=True)

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        normal = data_li[i]["mean"].iloc[0]
        trial_data = np.clip((data_li[i]["mean"]-normal)/normal, None, 0.5)
        best_data = np.clip((data_li[i]["best"]-normal)/normal, None, 0.5)
        ax.scatter(data_li[i]["trial_index"], trial_data, marker="x", label="trial")
        ax.plot(data_li[i]["trial_index"], best_data, label="best param")
        ax.legend()
        ax.set_xlabel("trial index")
        ax.set_ylabel("obj")
        ax.set_title("objective v/s trials, interval={}".format(st_i))
        fig.savefig(join(path, "trialvsobj_int_{}.svg".format(st_i)), bbox_inches="tight", transparent=True)

def plot_gaussianProcess(config, ax_clients, param_in):

    param_name = param_in
    density = 100
    metric_name = "execution_time"
    path = join(getcwd(), "output", "gaussianProcess")
    makedirs(path, exist_ok=True)

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):

        ax_client = ax_clients[i]
        modelbridge = Generators.BOTORCH_MODULAR(
            experiment=ax_client.experiment,
            data=ax_client.experiment.fetch_data(),
        )

        param = modelbridge._search_space.parameters[param_name]
        param_range = (param.lower, param.upper)
        x_vals = np.linspace(param_range[0], param_range[1], density)
        fixed_values = {}
        obs_features = []
        for x in x_vals:
            params = fixed_values.copy()
            params[param_name] = x
            obs_features.append(ObservationFeatures(parameters=params))

        means, covs = modelbridge.predict(obs_features)
        y_vals = np.array(means[metric_name])
        y_std = np.sqrt(np.array(covs[metric_name][metric_name]))
        fig_gp, ax_gp = plt.subplots(figsize=(6, 2.5))
        ax_gp.plot(x_vals, y_vals, label="Mean Prediction", marker="o")
        ax_gp.fill_between(x_vals, y_vals - 1.96*y_std, y_vals + 1.96*y_std, alpha=0.3, label="95% CI")
        ax_gp.set_title("Gaussian Process - interval = {} - {}".format(st_i, param_name))
        ax_gp.set_xlabel("{}".format(param_name))
        ax_gp.set_ylabel("Execution Time (Mean)")
        ax_gp.legend()
        fig_gp.savefig(join(path, "gauProc_{}_interval_{}.svg".format(param_name, st_i)), bbox_inches="tight", transparent=True)

if __name__ == '__main__':

    # load settings
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    try:
        with open(config_file, "r") as cf:
            config = safe_load(cf)
    except Exception as e:
        print(e)

    ax_clients = []
    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = AxClient().load_from_json_file(join(f"{config['experiment']['exp_path']}", f"ax_client_int_{i}.json"))
        ax_clients.append(ax_client)
    
    makedirs(join(getcwd(), "output"), exist_ok=True)
    plot_trial_vs_base(config, ax_clients)
    plot_best_params(config, ax_clients)
    plot_trialvsobj(config, ax_clients)
    plot_gaussianProcess(config, ax_clients, "relTol")