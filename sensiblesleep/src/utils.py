from typing import Tuple, Union

import numpy as np
import pandas as pd


def get_bin_from_time(hour: Union[int, float]) -> int:
    """
    Function to determine the bin value corresponding to a given hour of the day.

    Parameters:
    hour (Union[int, float]): The hour of the day in 24-hour format (e.g., 0-23.99).

    Returns:
    int: The bin value representing a 15-minute interval corresponding to the hour.

    Raises:
    ValueError: If the hour value is out of the expected range.
    """
    # Validate the hour input
    if not (0 <= hour < 24):
        raise ValueError("Hour value out of expected range (must be between 0 and 24).")

    # Start time in hours (16:00 is 16 hours)
    start_hour = 16

    # Compute the bin value depending on the time range
    if start_hour <= hour < 24:
        # First range: between 16:00 to 24:00
        bin_value = int((hour - start_hour) * 4)
    elif 0 <= hour < start_hour:
        # Second range: between 00:00 to 16:00
        bin_value = int((hour + 8) * 4)
    else:
        raise ValueError("Hour value out of expected range.")

    return bin_value


def get_time_from_bin(bin_value) -> str:
    """
    Function to determine the hour and minute of the day corresponding to a given bin value.

    Parameters:
    bin_value (int): The bin value representing a 15-minute interval.

    Returns:
    str: The hour and minute in HH:MM format corresponding to the bin value.

    Raises:
    ValueError: If the bin value is out of the expected range.
    """
    if not (0 <= bin_value < 96):
        raise ValueError("Bin value out of expected range.")

    # Start time in hours and minutes (16:00 is 16 hours and 0 minutes)
    start_hour = 16
    start_minute = 0

    # Calculate total minutes from the start time
    total_minutes = bin_value * 15
    hour_offset = total_minutes // 60
    minute_offset = total_minutes % 60

    # Calculate the current hour and minute in 24-hour format
    current_hour = (start_hour + hour_offset) % 24
    current_minute = (start_minute + minute_offset) % 60

    return f"{current_hour:02}:{current_minute:02}"


def extract_user_data(df: pd.DataFrame, user: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Extract data for a specific user from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing user data.
    user (str): The user ID to filter the DataFrame.

    Returns:
    Tuple[int, np.ndarray, np.ndarray]: A tuple containing the number of unique days, an array of time of day values,
                                        and an array of event counts for the specified user.
    """
    # Filter DataFrame for the specific user
    user_df = df[df["user"] == user]

    # Extract relevant information
    n_days = user_df["date"].nunique()  # number of unique days
    hours = user_df["time"].values  # time of day

    # Extract event counts and ensure no NaNs or non-numeric values
    observed_event_counts = np.nan_to_num(user_df["event_count"].values).astype("int64")

    return n_days, hours, observed_event_counts
