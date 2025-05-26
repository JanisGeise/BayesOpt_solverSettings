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
import pickle
from ax.modelbridge.base import ObservationFeatures
from copy import deepcopy
from ax.core.parameter import ChoiceParameter, RangeParameter
from ax.plot.diagnostic import interact_cross_validation
from ax.modelbridge.cross_validation import cross_validate

def plot_trial_vs_base(config, ax_clients):

    with open("benchmark_data.pkl", "rb") as f:
        data_li = pickle.load(f)

    dur = float(config['optimization']['duration'])
    step = int(dur/float(config['optimization']['deltaT']))
    mean_li = []
    for base_timing in data_li:
        t_plot = base_timing.t.values[step-1::step] - step*0.001
        t_cum_plot = base_timing.t_cpu_cum.values[step-1::step]
        t_cum_plot = np.concatenate((t_cum_plot[:1], t_cum_plot[1:]-t_cum_plot[:-1]))/step
        mean_li.append(t_cum_plot)

    box_li = np.array(mean_li)
    mean_array = np.mean(np.stack(mean_li), axis=0)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    x_centers = t_plot+step*0.001*0.9/2
    selected_times = [x + step*0.001*0.9/2 for x in map(float,config["optimization"]["startTime"])]
    flierprops = dict(marker='o', markersize=1, markerfacecolor='red', linestyle='none', markeredgecolor="red")
    boxprops = dict(linewidth=0.5, color='black')
    selected_indices = [i for i, t in enumerate(x_centers) if np.isclose(t, selected_times, atol=1e-6).any()]
    filtered_box_li = box_li[:, selected_indices]
    filtered_x_centers = x_centers[selected_indices]

    ax.boxplot(filtered_box_li, positions=filtered_x_centers, widths=step*0.001*0.9, patch_artist=False, flierprops=flierprops, boxprops=boxprops)
    ax.bar(t_plot, mean_array, width=step*0.001*0.9, align="edge", color="skyblue")
    ax.set_xlim(0, 6)
    #ax.set_ylim(0, 0.25)
    ax.set_xlabel(r"$\tilde{t}$")
    ax.set_ylabel(r"$T_{{{stept}\Delta t}}$".format(stept = step))
    ax.set_title("Execution time for base case - {} time steps".format(step))
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    
    ax.set_xticklabels([1, 2, 3, 4, 5, 6])


    for i, st_i in enumerate(config["optimization"]["startTime"][:]):

        data = ax_clients[i].experiment.fetch_data().df["mean"].values
        if i==0:
            ax.scatter([float(st_i)]*len(data), data, marker="x", s=10, c="red", label="trials", linewidth=0.5, alpha=0.5)
        else:
            ax.scatter([float(st_i)]*len(data), data, marker="x", s=10, c="red", linewidth=0.5, alpha=0.5)
    ax.legend()
    path = join(config["evaluation"]["output_path"], "trial_vs_base")
    makedirs(path, exist_ok=True)
    fig.savefig(join(path, "execution_time_dt_opt.svg"), bbox_inches="tight", transparent=True)
    plt.close(fig)

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
    config_plotScope = config["evaluation"]["plots"]["bestParams"]["plotScope"]
    if config_plotScope["type"] == "selected":
        sel_li = config_plotScope["selectedParams"]
        if set(sel_li).issubset(set(li)):
            li = sel_li
        else:
            raise ValueError("Selected params not in BO params")
    plot_li = ['interval']
    plot_li.extend(li)
    plot_df = df[plot_li]

    path = join(config["evaluation"]["output_path"], "best_params")
    makedirs(path, exist_ok=True)

    for xi in li:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(plot_df["interval"], plot_df[xi], marker="o")
        ax1.set_title("Best parameter - {} - for different intervals".format(xi))
        ax1.set_xlabel("interval")
        ax1.set_ylabel(xi)
        fig1.tight_layout()
        fig1.savefig(join(path, "parallel_plot_best_params_{}.svg".format(xi)), bbox_inches="tight", transparent=True)
        plt.close(fig1)

def plot_trialvsobj(config, ax_clients):
    data_li = []

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = ax_clients[i]
        df = ax_client.experiment.fetch_data().df
        df["best"] = df["mean"].cummin()
        data_li.append(df)

    path = join(config["evaluation"]["output_path"], "trialvsobj")
    makedirs(path, exist_ok=True)

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        normal = data_li[i]["mean"].iloc[0]
        trial_data = np.clip((data_li[i]["mean"])/normal, None, 1.5)
        best_data = np.clip((data_li[i]["best"])/normal, None, 1.5)
        ax.scatter(data_li[i]["trial_index"], trial_data, marker="x", label="trial")
        ax.plot(data_li[i]["trial_index"], best_data, label="best param")
        ax.legend()
        ax.set_xlabel("trial index")
        ax.set_ylabel("obj (normalized)")
        ax.set_title("objective v/s trials, interval={}".format(st_i))
        fig.savefig(join(path, "trialvsobj_int_{}.svg".format(st_i)), bbox_inches="tight", transparent=True)
        plt.close(fig)

def plot_gaussianProcess(config, ax_clients):

    density = 100
    metric_name = "execution_time"
    path = join(config["evaluation"]["output_path"], "gaussianProcess")
    makedirs(path, exist_ok=True)

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):

        ax_client = ax_clients[i]
        modelbridge = Generators.BOTORCH_MODULAR(
            experiment=ax_client.experiment,
            data=ax_client.experiment.fetch_data(),
        )
        param_names = [p for p in ax_client.experiment.parameters.keys()]
        param_setting = config["evaluation"]["plots"]["gaussianProcess"]["param_setting"]
        if param_setting["type"] == "optimal":
            slice_values = ax_client.get_best_parameters()[0]
        elif param_setting["type"] == "fixed":
            slice_values = param_setting["fixed_values"]
        else:
            raise ValueError("Unknown param_setting type")

        fixed_features = ObservationFeatures(parameters=slice_values)

        to_plot = []
        config_plotScope = config["evaluation"]["plots"]["gaussianProcess"]["plotScope"]
        if config_plotScope["type"] == "all":
            to_plot = param_names
        elif config_plotScope["type"] == "selected":
            to_plot = config_plotScope["selectedParams"]
        else:
            raise ValueError("Unknown plotScope type")

        for param_name in to_plot:
            param = modelbridge._search_space.parameters[param_name]

            if isinstance(param, ChoiceParameter):
                grid = param.values  # categorical grid: list of discrete values
            elif isinstance(param, RangeParameter):
                param_range = (param.lower, param.upper)
                grid = np.linspace(param_range[0], param_range[1], density)
            else:
                raise ValueError("Unsupported parameter type")
            
            fixed_values = slice_values.copy()
            for name, parame in modelbridge._search_space.parameters.items():
                if name not in fixed_values:
                    fixed_values[name] = ax_client.get_best_parameters()[0][name]
            prediction_features = []
            for x in grid:
                if fixed_features is None:
                    raise ValueError("Expected fixed_features to be non-None")
                predf = deepcopy(fixed_features)
                predf.parameters = fixed_values.copy()
                predf.parameters[param_name] = x
                prediction_features.append(predf)

            f, cov = modelbridge.predict(prediction_features)
            y_vals = np.array(f[metric_name])
            y_std = np.sqrt(np.array(cov[metric_name][metric_name]))
            # Plotting
            fig_gp, ax_gp = plt.subplots(figsize=(6, 2.5))
            ax_gp.plot(grid, y_vals, label="Mean Prediction", marker="o", markersize=1)
            ax_gp.fill_between(grid, y_vals - 1.96 * y_std, y_vals + 1.96 * y_std, alpha=0.3, label="95% CI")
            ax_gp.set_title(f"Gaussian Process - interval = {st_i} - {param_name}")
            ax_gp.set_xlabel(param_name)
            ax_gp.set_ylabel("Execution Time (Mean)")
            ax_gp.legend()
            if isinstance(param, ChoiceParameter):
                for label in ax_gp.get_xticklabels():
                    label.set_rotation(30)
                    label.set_horizontalalignment('right')
            fig_gp.tight_layout()
            fig_gp.savefig(join(path, f"gauProc_{param_name}_interval_{st_i}.svg"),
                           bbox_inches="tight", transparent=True)
            
            plt.close(fig_gp)

def plot_featureImportance(config, ax_clients):
    path = join(config["evaluation"]["output_path"], "featureImportance")
    makedirs(path, exist_ok=True)

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = ax_clients[i]
        modelbridge = Generators.BOTORCH_MODULAR(
            experiment=ax_client.experiment,
            data=ax_client.experiment.fetch_data(),
        )
        importances = modelbridge.feature_importances("execution_time")
        sorted_items = sorted(importances.items(), key=lambda x: -x[1])
        params, values = zip(*sorted_items)
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.bar(params, values, color='skyblue')
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance - interval={}".format(st_i))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            
        fig.tight_layout()
        fig.savefig(join(path, f"featureImportance_interval_{st_i}.svg"),
                           bbox_inches="tight", transparent=True)
        plt.close(fig)

def plot_crossValidation(config, ax_clients):
    path = join(config["evaluation"]["output_path"], "crossValidation")
    makedirs(path, exist_ok=True)

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = ax_clients[i]
        modelbridge = Generators.BOTORCH_MODULAR(
            experiment=ax_client.experiment,
            data=ax_client.experiment.fetch_data(),
        )
        cv_results = cross_validate(modelbridge)
        y_true = []
        y_pred = []
        y_std = []

        for cv_result in cv_results:
            observed_value = cv_result.observed.data.means[0]
            predicted_value = cv_result.predicted.means[0]
            variance = cv_result.predicted.covariance[0][0]
            std_dev = np.sqrt(variance)

            y_true.append(observed_value)
            y_pred.append(predicted_value)
            y_std.append(std_dev)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_std = np.array(y_std)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        
        ax.errorbar(y_true, y_pred, yerr=1.96 * y_std, fmt='o', capsize=3, label='Prediction (95% CI)', markersize = 2)
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal: y = x')

        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title("Cross Validation, interval={}".format(st_i))
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(join(path, f"crossValidation_interval_{st_i}.svg"),
                           bbox_inches="tight", transparent=True)
        plt.close(fig)

if __name__ == '__main__':

    # load settings
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    try:
        with open(config_file, "r") as cf:
            config = safe_load(cf)
    except Exception as e:
        print(e)
    
    config_eval = config['evaluation']
    plt.style.use(config_eval["plot_attributes"]["style_sheet"])
    plt.rcParams.update({
        "text.usetex": False
    })
    plt.rcParams.update({'font.size': 8})
    
    ax_clients = []
    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = AxClient().load_from_json_file(join(f"{config_eval['data_path']}", f"ax_client_int_{i}.json"))
        ax_clients.append(ax_client)
    
    makedirs(config_eval["output_path"], exist_ok=True)
    if config_eval["plots"]["trialVsBase"]["isReq"]:
        plot_trial_vs_base(config, ax_clients)
    if config_eval["plots"]["bestParams"]["isReq"]:
        plot_best_params(config, ax_clients)
    if config_eval["plots"]["trialVsObj"]["isReq"]:
        plot_trialvsobj(config, ax_clients)
    if config_eval["plots"]["gaussianProcess"]["isReq"]:
        plot_gaussianProcess(config, ax_clients)
    if config_eval["plots"]["featureImportance"]["isReq"]:
        plot_featureImportance(config, ax_clients)
    if config_eval["plots"]["crossValidation"]["isReq"]:
        plot_crossValidation(config, ax_clients)        