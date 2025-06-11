import sys
from os.path import join
from yaml import safe_load
from ax.service.ax_client import AxClient
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from os import makedirs
from ax.core.observation import ObservationFeatures
from ax.modelbridge.registry import Generators
from ax.modelbridge.base import ObservationFeatures
from copy import deepcopy
from ax.core.parameter import ChoiceParameter, RangeParameter
from ax.modelbridge.cross_validation import cross_validate
import os
import matplotlib as mpl
from ax.analysis.plotly.parallel_coordinates import ParallelCoordinatesPlot


def plot_trial_vs_base(config, ax_clients):
    """
    Plot base-case execution time vs. optimization trial results.

    This function compares cumulative CPU execution times from a set of
    baseline simulations to execution times from trials run during a
    Bayesian optimization loop. The base data is summarized with a bar
    plot and boxplot, while trial performance is plotted as scatter points.

    Parameters
    ----------
    config : dict
        Same configuration dictionary used for the Bayesian optimization run,
        which also contains the settings for evaluation. It must include keys
        like:
            - evaluation.benchmark_path : Path to base-case run folders.
            - evaluation.output_path : Directory to save the generated plot.
    ax_clients : list of AxClient
        List of AxClient instances, each representing optimization runs for
        different intervals.

    Returns
    -------
    None
        This function saves the plot to folder named "trial_vs_base" in the
        output directory and does not return any value.
    """
    if "benchmark_path" not in config.get("evaluation", {}):
        raise KeyError("Missing benchmark_path")
    fol = config["evaluation"]["benchmark_path"]
    folder = os.listdir(fol)
    data_li = []
    for file in folder:
        base_timing = read_csv(
            os.path.join(fol, file),
            header=None,
            sep=r"\s+",
            skiprows=1,
            usecols=[0, 1],
            names=["t", "t_cpu_cum"],
        )
        data_li.append(base_timing)
    dur = float(config["optimization"]["duration"])
    step = int(dur / float(config["optimization"]["deltaT"]))
    mean_li = []
    for base_timing in data_li:
        t_plot = base_timing.t.values[step - 1 :: step] - step * 0.001
        t_cum_plot = base_timing.t_cpu_cum.values[step - 1 :: step]
        t_cum_plot = (
            np.concatenate((t_cum_plot[:1], t_cum_plot[1:] - t_cum_plot[:-1])) / step
        )
        mean_li.append(t_cum_plot)

    box_li = np.array(mean_li)
    mean_array = np.mean(np.stack(mean_li), axis=0)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    x_centers = t_plot + step * 0.001 * 0.9 / 2
    selected_times = [
        x + step * 0.001 * 0.9 / 2
        for x in map(float, config["optimization"]["startTime"])
    ]
    flierprops = dict(
        marker="o",
        markersize=1,
        markerfacecolor="red",
        linestyle="none",
        markeredgecolor="red",
    )
    boxprops = dict(linewidth=0.5, color="black")
    selected_indices = [
        i
        for i, t in enumerate(x_centers)
        if np.isclose(t, selected_times, atol=1e-6).any()
    ]
    filtered_box_li = box_li[:, selected_indices]
    filtered_x_centers = x_centers[selected_indices]

    ax.boxplot(
        filtered_box_li,
        positions=filtered_x_centers,
        widths=step * 0.001 * 0.9,
        patch_artist=False,
        flierprops=flierprops,
        boxprops=boxprops,
    )

    ax.bar(t_plot, mean_array, width=step * 0.001 * 0.9, align="edge", color="skyblue")
    xlim_upper = float(config["simulation"]["startTime"]) + float(
        config["simulation"]["duration"]
    )
    ax.set_xlim(0, xlim_upper)
    ax.set_xlabel(r"$\tilde{t}$")
    ax.set_ylabel(r"$T_{{{stept}\Delta t}}$".format(stept=step))
    ax.set_title("Execution time for base case - {} time steps".format(step))
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([1, 2, 3, 4, 5, 6])

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):

        data = ax_clients[i].experiment.fetch_data().df["mean"].values
        if i == 0:
            ax.scatter(
                [float(st_i) + step * 0.001 * 0.9 / 2] * len(data),
                data,
                marker="x",
                s=10,
                c="red",
                label="trials",
                linewidth=0.5,
                alpha=0.5,
            )
        else:
            ax.scatter(
                [float(st_i) + step * 0.001 * 0.9 / 2] * len(data),
                data,
                marker="x",
                s=10,
                c="red",
                linewidth=0.5,
                alpha=0.5,
            )
    ax.legend()
    path = join(config["evaluation"]["output_path"], "trial_vs_base")
    makedirs(path, exist_ok=True)
    fig.savefig(
        join(path, "execution_time_dt_opt.svg"), bbox_inches="tight", transparent=True
    )
    plt.close(fig)


def plot_best_params(config, ax_clients):
    """
    Plot the best parameters across different optimization intervals.

    This function extracts the top-N best parameters obtained from Bayesian optimization
    for each specified interval (start time), and generates line plots showing how
    the best values of each parameter evolves with the interval. The user can choose
    to plot only a subset of parameters through the configuration. The markers are colored
    to signify the execution times and the ranking of the best parameter sets is done by
    coloring the lines.

    Parameters
    ----------
    config : dict
        Same configuration dictionary used for the Bayesian optimization run,
        which also contains the settings for evaluation. It must include keys
        like:
            - evaluation.output_path : Path to save the generated plots.
            - evaluation.plots.best_params.plot_scope : Dict with keys:
                - type : "all" or "selected"
                - selected_params : List of parameters to plot if type is "selected"
            - evaluation.plots.best_params.top_N : Number of best trials to plot
    ax_clients : list of AxClient
        List of AxClient instances, each representing optimization runs for
        different intervals.

    Returns
    -------
    None
        This function saves the plot to folder named "best_params" in the
        output directory and does not return any value.
    """
    dic_best = {}
    rows = []
    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = ax_clients[i]
        df_trials = ax_client.get_trials_data_frame()
        df_trials = df_trials.query("trial_status == 'COMPLETED'")
        df_sorted = df_trials.sort_values(by="execution_time", ascending=True).head(5)
        dic_best[float(st_i)] = ax_client.get_best_parameters()
        for rank, trial_index in enumerate(df_sorted.trial_index, start=1):
            params = ax_client.experiment.trials[trial_index].arm.parameters.copy()
            params.update(
                {
                    "interval": float(st_i),
                    "rank": rank,
                    "execution_time": df_sorted["execution_time"].iloc[rank - 1],
                }
            )
            rows.append(params)
    df = DataFrame(rows)
    param_cols = [
        c for c in df.columns if c not in ("interval", "rank", "execution_time")
    ]
    cfg_scope = config["evaluation"]["plots"]["best_params"]["plot_scope"]
    if cfg_scope["type"] == "selected":
        sel = cfg_scope["selected_params"]
        if not set(sel).issubset(param_cols):
            raise ValueError("Selected params not in BO params")
        param_cols = sel
    cmap = mpl.colormaps.get_cmap("tab10")
    path = join(config["evaluation"]["output_path"], "best_params")
    makedirs(path, exist_ok=True)
    if "top_N" not in config.get("evaluation", {}).get("plots", {}).get(
        "best_params", {}
    ):
        raise ValueError("top_N parameter not defined")
    top_n = config["evaluation"]["plots"]["best_params"]["top_N"]
    max_z = top_n + 10
    for p in param_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        for rank in range(1, top_n + 1):
            slice_i = df[df["rank"] == rank]
            x = slice_i["interval"]
            y = slice_i[p]
            c = slice_i["execution_time"]
            ax.plot(
                x,
                y,
                color=cmap((rank - 1) % cmap.N),
                linewidth=1,
                alpha=0.7,
                label=f"rank {rank}",
                zorder=max_z - rank,
            )

            scatter = ax.scatter(
                x,
                y,
                c=c,
                cmap="viridis",
                s=70,
                edgecolors="k",
                zorder=max_z - rank + 0.5,
            )

        ax.set_title(f"Ranking best params - {p} across optimisation intervals")
        ax.set_xlabel("interval")
        ax.set_ylabel(p)
        ax.legend(title="trial rank")

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Execution Time (s)")

        fig.tight_layout()
        fig.savefig(
            join(path, f"best_param_plot_top{top_n}_{p}.svg"),
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


def plot_trial_vs_obj(config, ax_clients):
    """
    Plot objective values across trials for different optimization intervals.

    This function visualizes how the objective metric (e.g., execution time)
    evolves across Bayesian optimization trials. For each interval defined by
    a start time, it generates a scatter plot of trial objective values and a
    line plot showing the best (minimum) value found up to each trial. The
    objective values are normalized relative to the first trial.

    Parameters
    ----------
    config : dict
        Same configuration dictionary used for the Bayesian optimization run,
        which also contains the settings for evaluation. It must include keys
        like:
            - evaluation.output_path : Directory to save the generated plot.
    ax_clients : list of AxClient
        List of AxClient instances, each representing optimization runs for
        different intervals.

    Returns
    -------
    None
        This function saves the plot to folder named "trial_vs_obj" in the
        output directory and does not return any value.
    """
    data_li = []
    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = ax_clients[i]
        df = ax_client.experiment.fetch_data().df
        df["best"] = df["mean"].cummin()
        data_li.append(df)

    path = join(config["evaluation"]["output_path"], "trial_vs_obj")
    makedirs(path, exist_ok=True)

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        normal = data_li[i]["mean"].iloc[0]
        trial_data = np.clip((data_li[i]["mean"]) / normal, None, 1.5)
        best_data = np.clip((data_li[i]["best"]) / normal, None, 1.5)
        ax.scatter(data_li[i]["trial_index"], trial_data, marker="x", label="trial")
        ax.plot(data_li[i]["trial_index"], best_data, label="best param")
        ax.legend()
        ax.set_xlabel("trial index")
        ax.set_ylabel("obj (normalized)")
        ax.set_title("objective v/s trials, interval={}".format(st_i))
        fig.savefig(
            join(path, "trialvsobj_int_{}.svg".format(st_i)),
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


def plot_gaussian_process(config, ax_clients):
    """
    Plot Gaussian Process (GP) predictions for different parameters across
    intervals.

    This function visualizes the mean prediction and uncertainty of the Gaussian
    Process model learned during Bayesian Optimization (via BoTorch/Ax) for
    each optimization interval. For each selected parameter, predictions are
    made over a grid (for continuous variables) or discrete values (for
    categorical variables), while other parameters are handled using fixed values,
    optimal values, or marginalized over all observed trials.

    Parameters
    ----------
    config : dict
        Same configuration dictionary used for the Bayesian optimization run,
        which also contains the settings for evaluation. It must include keys
        like:
            - evaluation.output_path : Path where plots will be saved.
            - evaluation.plots.gaussian_process.param_setting :
                Strategy for fixing parameters not currently being plotted:
                - type : "optimal", "fixed" or "marginalization"
                    * "optimal" : Uses the best parameter set found via
                                  optimization for predictions.
                    * "fixed" : Uses a predefined fixed parameter dictionary,
                                supplemented with optimal values for missing keys.
                    * "marginalization" : Uses all observed parameter sets
                                         (from all trials), and predictions
                                         are averaged over these to marginalize
                                         the effect of non-plotted parameters.
            - evaluation.plots.gaussian_process.plot_scope :
                Scope of parameters to include in plots:
                - type : "all" or "selected"
                - selected_params : Names of parameters to include
                                    in plots (if type is "selected").
    ax_clients : list of AxClient
        List of AxClient instances, each representing optimization runs for
        different intervals.

    Returns
    -------
    None
        This function saves the plot to folder named "gaussian_process" in the
        output directory and does not return any value.

    Notes
    -----
    - For `RangeParameter`s (continuous), the mean and 95% confidence interval
      are plotted over a grid.
    - For `ChoiceParameter`s (categorical), a bar chart with error bars is used.
    - Observed trial results are overlaid as scatter points for context.
    """
    density = 100
    metric_name = "execution_time"
    path = join(config["evaluation"]["output_path"], "gaussian_process")
    makedirs(path, exist_ok=True)

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = ax_clients[i]
        modelbridge = Generators.BOTORCH_MODULAR(
            experiment=ax_client.experiment,
            data=ax_client.experiment.fetch_data(),
        )
        param_names = [p for p in ax_client.experiment.parameters.keys()]
        param_setting = config["evaluation"]["plots"]["gaussian_process"][
            "param_setting"
        ]
        if param_setting["type"] == "optimal":
            slice_values = [ax_client.get_best_parameters()[0]]
        elif param_setting["type"] == "fixed":
            slice_values = [param_setting["fixed_values"]]
            for name, parame in modelbridge._search_space.parameters.items():
                if name not in slice_values[0]:
                    slice_values[0][name] = ax_client.get_best_parameters()[0][name]
        elif param_setting["type"] == "marginalization":
            sobol_model = Generators.SOBOL(experiment=ax_client.experiment)
            num_marginal_samples = config["evaluation"]["plots"][
                "gaussian_process"
            ].get("n_marginal_samples", 50)
            sobol_generator_run = sobol_model.gen(n=num_marginal_samples)
            slice_values = [arm.parameters for arm in sobol_generator_run.arms]
        else:
            raise ValueError("Unknown param_setting type")

        config_gp_scope = config["evaluation"]["plots"]["gaussian_process"][
            "plot_scope"
        ]
        to_plot = []
        if config_gp_scope["type"] == "all":
            to_plot = param_names
        elif config_gp_scope["type"] == "selected":
            to_plot = config_gp_scope["selected_params"]
        else:
            raise ValueError("Unknown plot_scope type")
        data = ax_client.experiment.fetch_data()
        df = data.df

        for param_name in to_plot:
            param = modelbridge._search_space.parameters[param_name]
            obs_x = []
            obs_y = []
            for trial in ax_client.experiment.trials.values():
                arm = trial.arm
                param_val = arm.parameters[param_name]
                obs_x.append(param_val)
                obs_y.append(df[df["arm_name"] == arm.name]["mean"].values[0])

            if isinstance(param, ChoiceParameter):
                grid = param.values
            elif isinstance(param, RangeParameter):
                param_range = (param.lower, param.upper)
                grid = np.linspace(param_range[0], param_range[1], density)
            else:
                raise ValueError("Unsupported parameter type")

            prediction_features = []
            for x in grid:
                for sample in slice_values:
                    pred_params = deepcopy(sample)
                    pred_params[param_name] = x
                    prediction_features.append(
                        ObservationFeatures(parameters=pred_params)
                    )

            means, covs = modelbridge.predict(prediction_features)
            mean_array = np.array(means[metric_name]).reshape((len(grid), -1))
            std_array = np.sqrt(np.array(covs[metric_name][metric_name])).reshape(
                (len(grid), -1)
            )
            y_vals = mean_array.mean(axis=1)
            y_std = std_array.mean(axis=1)
            # Plotting
            fig_gp, ax_gp = plt.subplots(figsize=(6, 2.5))
            if isinstance(param, RangeParameter):
                ax_gp.plot(grid, y_vals, label="Mean Prediction")
                ax_gp.fill_between(
                    grid,
                    y_vals - 1.96 * y_std,
                    y_vals + 1.96 * y_std,
                    alpha=0.3,
                    label="95% CI",
                )
                ax_gp.scatter(
                    obs_x, obs_y, color="blue", label="Observed Trials", marker="x", s=2
                )
                low_lim, up_lim = config["optimization"]["gamg"][param_name]["bounds"]
                ax_gp.set_xlim(low_lim, up_lim)
            elif isinstance(param, ChoiceParameter):
                ax_gp.bar(grid, y_vals, color="skyblue", label="Mean Prediction")
                ax_gp.errorbar(
                    grid,
                    y_vals,
                    yerr=1.96 * y_std,
                    fmt="o",
                    color="r",
                    capsize=2,
                    label="95% CI",
                )
                ax_gp.scatter(
                    obs_x, obs_y, color="blue", label="Observed Trials", marker="x", s=2
                )
                for label in ax_gp.get_xticklabels():
                    label.set_rotation(30)
                    label.set_horizontalalignment("right")
            else:
                raise KeyError("Paramter type not defined")
            type = param_setting["type"]
            ax_gp.set_title(f"Gaussian Process - interval = {st_i} - {type}")
            ax_gp.set_xlabel(param_name)
            ax_gp.set_ylabel("Execution Time (Mean)")
            ax_gp.legend()
            fig_gp.savefig(
                join(path, f"gauProc_{param_name}_interval_{st_i}.svg"),
                bbox_inches="tight",
                transparent=True,
            )

            plt.close(fig_gp)


def plot_feature_importance(config, ax_clients):
    """
    Plot feature importance of parameters for each optimization interval.

    This function computes and visualizes the relative importance of input
    parameters to the objective metric (execution time) as estimated by the
    Gaussian Process model trained during Bayesian Optimization. It handles both
    continuous and categorical parameters.

    Parameters
    ----------
    config : dict
        Same configuration dictionary used for the Bayesian optimization run,
        which also contains the settings for evaluation. It must include keys
        like:
            - evaluation.output_path : Directory to save the generated plot.
    ax_clients : list of AxClient
        List of AxClient instances, each representing optimization runs for
        different intervals.

    Returns
    -------
    None
        This function saves the plot to folder named "feature_importance" in the
        output directory and does not return any value.
    """

    path = join(config["evaluation"]["output_path"], "feature_importance")
    makedirs(path, exist_ok=True)

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = ax_clients[i]
        modelbridge = Generators.BOTORCH_MODULAR(
            experiment=ax_client.experiment,
            data=ax_client.experiment.fetch_data(),
        )
        param_names = [p for p in ax_client.experiment.parameters.keys()]
        importances = modelbridge.feature_importances("execution_time")
        aggregated_importances = {}

        for param in param_names:
            if isinstance(modelbridge._search_space.parameters[param], ChoiceParameter):
                total = sum(
                    value for key, value in importances.items() if key.startswith(param)
                )
                aggregated_importances[param] = total
            else:
                if param in importances:
                    aggregated_importances[param] = importances[param]

        sorted_items = sorted(aggregated_importances.items(), key=lambda x: -x[1])
        params, values = zip(*sorted_items)
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.bar(params, values, color="skyblue")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance - interval={}".format(st_i))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")

        fig.tight_layout()
        fig.savefig(
            join(path, f"featureImportance_interval_{st_i}.svg"),
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


def plot_cross_validation(config, ax_clients):
    """
    Perform and plot cross-validation results for each optimization interval.

    This function evaluates the generalization performance of the trained
    Gaussian Process (GP) surrogate model using leave-one-out cross-validation
    on observed data. It compares predicted values with actual (observed)
    objective values (execution time) and visualizes the results with
    prediction error bars and ideal y=x reference lines.

    Parameters
    ----------
    config : dict
        Same configuration dictionary used for the Bayesian optimization run,
        which also contains the settings for evaluation. It must include keys
        like:
            - evaluation.output_path : Directory to save the generated plot.
    ax_clients : list of AxClient
        List of AxClient instances, each representing optimization runs for
        different intervals.

    Returns
    -------
    None
        This function saves the plot to folder named "cross_validation" in the
        output directory and does not return any value.
    """

    path = join(config["evaluation"]["output_path"], "cross_validation")
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

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.errorbar(
            y_true,
            y_pred,
            yerr=1.96 * y_std,
            fmt="o",
            capsize=3,
            label="Prediction (95% CI)",
            markersize=2,
        )
        ax.plot(
            [min(y_true), max(y_true)],
            [min(y_true), max(y_true)],
            "r--",
            label="Ideal: y = x",
        )
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title("Cross Validation, interval={}".format(st_i))
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(
            join(path, f"crossValidation_interval_{st_i}.svg"),
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)

def plot_parallel_coordinates(config, ax_clients):
    """
    Generate a parallel coordinates plot to visualize high-dimensional data.

    This plot is useful for understanding trends, patterns, and trade-offs
    between multiple input features across samples or trials.

    Parameters
    ----------
    config : dict
        Same configuration dictionary used for the Bayesian optimization run,
        which also contains the settings for evaluation. It must include keys
        like:
            - evaluation.output_path : Directory to save the generated plot.
    ax_clients : list of AxClient
        List of AxClient instances, each representing optimization runs for
        different intervals.

    Returns
    -------
    None
        This function saves the plot to folder named "parallel_coordinates" in the
        output directory and does not return any value.
    """

    path = join(config["evaluation"]["output_path"], "parallel_coordinates")
    makedirs(path, exist_ok=True)

    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = ax_clients[i]
        analysis = ParallelCoordinatesPlot()
        cards = analysis.compute(
            experiment=ax_client._experiment,
            generation_strategy=ax_client._generation_strategy,
            adapter=None,
        )
        fig = cards[0].get_figure()
        fig.update_layout(
            title={
                "text": "Parallel coordinates plot, interval={}".format(st_i),
                "x": 0.5,
                "xanchor": "center",
                "y": 0.05,
                "yanchor": "top",
            }
        )
        fig.write_image(
            join(path, f"parallel_coordinates_interval_{st_i}.png"),
            width=1300,
            height=800,
        )


if __name__ == "__main__":

    # load settings
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    try:
        with open(config_file, "r") as cf:
            config = safe_load(cf)
    except Exception as e:
        print(e)

    config_eval = config["evaluation"]
    plt.style.use(config_eval["plot_attributes"]["style_sheet"])
    plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({"font.size": 8})

    ax_clients = []
    for i, st_i in enumerate(config["optimization"]["startTime"][:]):
        ax_client = AxClient().load_from_json_file(
            join(f"{config['experiment']['exp_path']}", f"ax_client_int_{i}.json")
        )
        ax_clients.append(ax_client)

    makedirs(config_eval["output_path"], exist_ok=True)
    if config_eval["plots"]["trial_vs_base"]["isReq"]:
        plot_trial_vs_base(config, ax_clients)
    if config_eval["plots"]["best_params"]["isReq"]:
        plot_best_params(config, ax_clients)
    if config_eval["plots"]["trial_vs_obj"]["isReq"]:
        plot_trial_vs_obj(config, ax_clients)
    if config_eval["plots"]["gaussian_process"]["isReq"]:
        plot_gaussian_process(config, ax_clients)
    if config_eval["plots"]["feature_importance"]["isReq"]:
        plot_feature_importance(config, ax_clients)
    if config_eval["plots"]["cross_validation"]["isReq"]:
        plot_cross_validation(config, ax_clients)
    if config_eval["plots"]["parallel_coordinates"]["isReq"]:
        plot_parallel_coordinates(config, ax_clients)
