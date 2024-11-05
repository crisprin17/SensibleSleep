import json
import os
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

script_dir = os.path.dirname(
    os.path.abspath(__file__)
)  # Absolute path to the script directory
CONFIG_PATH = os.path.join(script_dir, "data_config.json")  # Path to data_config.json


def generate_negative_binomial_data(
    user_id,
    start_date,
    n_days,
    avg_screen_ons,
    minutes_per_bin,
    overdispersion_factor=2,
):
    data = []
    nb_mean = avg_screen_ons

    # Calculate variance as overdispersion_factor * mean
    nb_variance = overdispersion_factor * nb_mean

    # Calculate dispersion parameter based on chosen variance
    nb_dispersion = (nb_mean**2) / (nb_variance - nb_mean)

    for day in range(n_days):
        current_date = start_date + timedelta(days=day)

        # Generate total events using a negative binomial distribution
        # The mean of the negative binomial is `nb_mean` and variance is `nb_mean + nb_mean^2 / nb_dispersion`
        total_events = np.random.negative_binomial(
            nb_dispersion, nb_dispersion / (nb_dispersion + nb_mean)
        )

        n_bins = 24 * 60 // minutes_per_bin  # Number of X-minute bins in a day
        screen_events = np.zeros(n_bins)

        # Define bins for daytime (higher activity) and nighttime (lower activity)
        day_bins = list(range(0, 28)) + list(
            range(56, 96)
        )  # Daytime: 16:00 to 07:00 next day
        night_bins = list(range(28, 56))  # Nighttime: 07:00 to 16:00

        # Allocate events to daytime bins (99.5% of events during the day)
        day_event_count = int(total_events * 0.995)
        for _ in range(day_event_count):
            bin_idx = np.random.choice(day_bins)
            screen_events[bin_idx] += 1

        # Allocate remaining events to nighttime bins
        night_event_count = total_events - day_event_count
        for _ in range(night_event_count):
            bin_idx = np.random.choice(night_bins)
            screen_events[bin_idx] += 1

        # Log the number of events for each 15-minute bin
        for bin_idx in range(n_bins):
            hour = (bin_idx // 4 + 16) % 24  # Adjust hour to start at 16:00
            minute = (bin_idx % 4) * 15  # 15-minute intervals

            # Append the result for this user, day, and bin
            data.append(
                [
                    user_id,
                    current_date.strftime("%Y-%m-%d"),
                    f"{hour:02d}:{minute:02d}",
                    int(screen_events[bin_idx]),
                ]
            )

    return data


def generate_user_data(
    user_id: str,
    start_date: datetime,
    n_days: int,
    minutes_per_bin: int,
    avg_screen_ons: float,
) -> List[List[str]]:
    """
    Function to generate synthetic user data for screen-on events.

    Parameters:
    user_id (str): Unique identifier for the user.
    start_date (datetime): The start date for data generation.
    n_days (int): Number of days to generate data for.
    minutes_per_bin (int): Number of minutes per time bin (e.g., 15 for 15-minute intervals).
    avg_screen_ons (float): Average number of screen-on events per day.

    Returns:
    List[List[str]]: Generated data in the form of a list of lists containing user ID, date, time, and event count.
    """
    data = []
    for day in range(n_days):
        current_date = start_date + timedelta(days=day)

        # Generate random screen-on events across bins, summing to approx. avg_screen_ons events per day
        total_events = np.random.poisson(
            avg_screen_ons
        )  # Poisson-distributed around avg_screen_ons
        n_bins = 24 * 60 // minutes_per_bin  # Number of X-minute bins in a day
        screen_events = np.zeros(n_bins)

        # Define bins for daytime (higher activity) and nighttime (lower activity)
        day_bins = list(range(0, 28)) + list(
            range(56, 96)
        )  # Daytime: 16:00 to 07:00 next day
        night_bins = list(range(28, 56))  # Nighttime: 07:00 to 16:00

        # Allocate events to daytime bins (99.5% of events during the day)
        day_event_count = int(total_events * 0.995)
        for _ in range(day_event_count):
            bin_idx = np.random.choice(day_bins)
            screen_events[bin_idx] += 1

        # Allocate remaining events to nighttime bins
        night_event_count = total_events - day_event_count
        for _ in range(night_event_count):
            bin_idx = np.random.choice(night_bins)
            screen_events[bin_idx] += 1

        # Log the number of events for each 15-minute bin
        for bin_idx in range(n_bins):
            hour = (bin_idx // 4 + 16) % 24  # Adjust hour to start at 16:00
            minute = (bin_idx % 4) * 15  # 15-minute intervals

            # Append the result for this user, day, and bin
            data.append(
                [
                    user_id,
                    current_date.strftime("%Y-%m-%d"),
                    f"{hour:02d}:{minute:02d}",
                    int(screen_events[bin_idx]),
                ]
            )

    return data


def generate_synthetic_data(use_negative_binomial: bool = False) -> pd.DataFrame:
    """
    Function to generate synthetic data for multiple users over a number of days.

    Returns:
    pd.DataFrame: A DataFrame containing the generated synthetic data with
                columns 'user', 'date', 'time', and 'event_count'.
    """
    DATA_CONFIG = json.load(open(CONFIG_PATH))
    n_users = DATA_CONFIG["data"]["n_users"]
    n_days = DATA_CONFIG["data"]["n_days"]
    minutes_per_bin = DATA_CONFIG["data"]["minutes_per_bin"]
    avg_screen_ons = DATA_CONFIG["data"]["avg_screen_ons"]
    start_date = datetime(2024, 9, 1)  # Starting date for the dataset

    all_data = []
    for user_idx in range(n_users):
        user_id = f"user_{user_idx + 1:03d}"
        if use_negative_binomial:
            user_data = generate_negative_binomial_data(
                user_id=user_id,
                start_date=start_date,
                n_days=n_days,
                avg_screen_ons=avg_screen_ons,
                minutes_per_bin=minutes_per_bin,
            )
        else:
            user_data = generate_user_data(
                user_id=user_id,
                start_date=start_date,
                n_days=n_days,
                minutes_per_bin=minutes_per_bin,
                avg_screen_ons=avg_screen_ons,
            )
        all_data.extend(user_data)

    # Create a DataFrame from the generated data
    columns = ["user", "date", "time", "event_count"]
    df = pd.DataFrame(all_data, columns=columns)
    return df
