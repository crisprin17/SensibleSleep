from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from . import utils


def plot_screen_events(observed_event_counts: np.ndarray, n_days: int) -> None:
    """
    Plot user-specific screen event data over a 24-hour period.

    Parameters:
    observed_event_counts (np.ndarray): Array of observed screen event counts for each time bin.
    n_days (int): Number of days in the dataset.

    Returns:
    None
    """
    hours = np.arange(0, 96)
    hours = np.tile(hours, n_days)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bar chart of event counts
    ax.bar(hours, observed_event_counts, width=0.8)

    # Plot the sleep and wake times with red dashed lines
    ax.axvline(
        x=utils.get_bin_from_time(23),
        color="red",
        linestyle="--",
        linewidth=2,
        label="tsleep (23:00)",
    )
    ax.axvline(
        x=utils.get_bin_from_time(6),
        color="red",
        linestyle="--",
        linewidth=2,
        label="tawake (06:00)",
    )

    ax.set_xlabel("Hour of the Day", fontsize=12)
    ax.set_ylabel("Number of Screen Events", fontsize=12)

    # Set x-axis ticks to show time in a readable format
    ax.set_xticks(np.arange(0, 96, 4))  # Set ticks for 96 bins
    ax.set_xticklabels(
        [f"{int(hour):02d}:00" for hour in np.arange(16, 16 + 24) % 24], rotation=45
    )

    # Annotating lambda values for awake and sleep periods
    ax.text(
        utils.get_bin_from_time(17),
        max(observed_event_counts) * 0.8,
        r"$\lambda_{awake}$",
        fontsize=14,
    )
    ax.text(
        utils.get_bin_from_time(2),
        max(observed_event_counts) * 0.8,
        r"$\lambda_{sleep}$",
        fontsize=14,
    )
    ax.text(
        utils.get_bin_from_time(9),
        max(observed_event_counts) * 0.8,
        r"$\lambda_{awake}$",
        fontsize=14,
    )

    # Annotating tsleep and tawake
    ax.text(
        utils.get_bin_from_time(23.5),
        max(observed_event_counts) - 2,
        r"$t_{sleep}$",
        fontsize=14,
        color="red",
    )
    ax.text(
        utils.get_bin_from_time(5.5),
        max(observed_event_counts) - 2,
        r"$t_{awake}$",
        fontsize=14,
        color="red",
    )

    # Add grid for clarity
    ax.grid(True)
    ax.legend()
    plt.title("Number of Screen Events and Sleep/Wake Switchpoints", fontsize=14)
    plt.show()


def plot_logp(map_models: Dict[str, xr.Dataset], map_colors: Dict[str, str]) -> None:
    """
    Function to plot the negative log posterior for each model.

    Parameters:
    map_models (Dict[str, xr.Dataset]): Dictionary mapping model names to
                their xarray Dataset containing posterior samples.
    map_colors (Dict[str, str]): Dictionary mapping model names to
                their respective colors for plotting.
    """
    # Now, plot the negative logp for each model
    plt.figure(figsize=(10, 6))
    for k, v in map_models.items():
        # Plot negative logp for each model
        if k == "hyper hyper":
            v = v.mean(dim=["chain"]).values.mean(axis=1)
        else:
            v = v.mean(dim=["chain"]).values
        plt.plot(-v, label=k, color=map_colors[k])

    # Customize plot
    plt.xlabel("Trace position", fontsize=12)
    plt.ylabel("Negative logp", fontsize=12)
    plt.title("Negative Log Posterior Comparison Across Models", fontsize=14)
    plt.legend()

    # Show the plot
    plt.show()


def plot_event_counts(df: pd.DataFrame, user: Any) -> None:
    """
    Function to plot the histogram of event counts grouped by user.

    Parameters:
    df (pd.DataFrame): DataFrame containing event data with
                columns 'user', 'time', and 'event_count'.
    user (Any): User identifier to filter the DataFrame for plotting.
    """

    # Plot the histogram of event counts grouped by user
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=df[df["user"] == user],
        x="time",
        weights="event_count",
        hue="user",
        multiple="stack",
        palette="Set1",
        bins=96,
    )

    # Customize the plot
    plt.title("Event Counts by Time Bin, Stacked by User", fontsize=14)
    plt.xlabel("Time Bin (15-min intervals over 24 hours)", fontsize=12)
    plt.ylabel("Total Event Count", fontsize=12)
    plt.xticks(
        ticks=np.arange(0, 96, 4),
        labels=[f"{int(hour)}:00" for hour in np.arange(16, 16 + 24) % 24],
        rotation=45,
    )

    # Show the plot
    plt.legend(title="User")
    plt.show()


def plot_posterior_sleep_wake(
    tsleep_samples: List[float], tawake_samples: List[float], n_bins: int = 96
) -> None:
    """
    Function to plot the aggregate posterior probability distributions of `tsleep` and `tawake` as histograms.

    Parameters:
    tsleep_samples (List[float]): List of posterior samples for sleep time (`tsleep`).
    tawake_samples (List[float]): List of posterior samples for wake time (`tawake`).
    n_bins (int): Number of bins for the histogram (default 96 for 15-minute intervals over a 24-hour period).
    """

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms for the posterior samples with transparency (alpha) to avoid overlap
    ax.hist(
        tsleep_samples,
        bins=n_bins,
        density=True,
        color="blue",
        alpha=0.6,
        label=r"$t_{sleep}$",
    )
    ax.hist(
        tawake_samples,
        bins=n_bins,
        density=True,
        color="green",
        alpha=0.6,
        label=r"$t_{awake}$",
    )

    # Add labels and legend
    ax.set_xlabel("Hour of the Day", fontsize=12)
    ax.set_ylabel("P(Hour of the Day)", fontsize=12)

    # Set x-axis ticks to display hours (e.g., 16:00, 17:00, etc.)
    hours = np.arange(16, 16 + 24) % 24
    ax.set_xticks(np.arange(0, n_bins, n_bins // 24))
    ax.set_xticklabels([f"{int(hour)}:00" for hour in hours], rotation=45)

    # Add legend
    ax.legend()

    # Set title and show the plot
    ax.set_title(
        "Aggregate Posterior Probability Distributions of $t_{sleep}$ and $t_{awake}$",
        fontsize=14,
    )
    plt.show()


def plot_DIC(
    reference_DIC,
    DIC_values,
    DIC_errors,
    model_names,
    reference_model: str = "pooled_pooled",
) -> None:
    """
    Function to plot the DIC relative to a reference model for each model.

    Parameters:
    reference_DIC (float): DIC value of the reference model.
    DIC_values (list[float]): List of DIC values for each model.
    dic_errors (list[float]): List of standard errors for the DIC values.
    reference_model (str): Name of the reference model for relative DIC comparison.
    """
    # Normalize DIC values relative to the reference DIC
    DIC_values = [x / reference_DIC for x in DIC_values]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(DIC_values))

    # Plot the bars with error bars
    ax.barh(
        y_pos,
        DIC_values,
        xerr=DIC_errors,
        color="blue",
        edgecolor="black",
        align="center",
    )
    ax.errorbar(DIC_values, y_pos, xerr=DIC_errors, fmt="none", ecolor="red", capsize=5)

    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name for name in model_names if name != reference_model])
    ax.set_xlabel(f"DIC relative to {reference_model}")
    ax.invert_yaxis()  # Reverse order to match example

    plt.show()


def plot_sleep_duration(sleep_durations: List[float]) -> None:
    """
    Function to plot the histogram of sleep durations.

    Parameters:
    sleep_durations (List[float]): List of sleep durations.
    """

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.hist(sleep_durations, bins=15, color="blue", edgecolor="black", density=True)

    # Set labels and title
    plt.xlabel("Average sleep duration (hours)")
    plt.ylabel("P(Average sleep duration)")
    plt.show()
