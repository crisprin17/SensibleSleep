# Synthetic Data Generation for SensibleSleep

## Data Requirements

### Users
- **Number of Users**: 126 users.
- **Sleep Tracking**: Each user must have at least 3 hours of tracked sleep per day.

### Date Range
- **Duration**: Data is generated for each user across a continuous period of 2 to 4 weeks. For simplicity, we assume 21 days of data.

### Screen Events
- **Sleep Hours**: Most screen-off events are expected during typical sleep hours, which are between **23:00 and 07:00**.
- **Real-World Interruptions**: Some random screen interactions (screen-on events) are recorded during sleep hours to simulate real-world interruptions.
- **Waking Hours**: Normal screen usage is tracked during waking hours, which are approximately from **07:00 to 23:00**.

### Sleep and Screen-On Patterns
- **Sleep Period**: Typically from **23:00 to 07:00**, with occasional screen-on events (about a **5% chance** of activation).
- **Daytime Usage**: During waking hours, the user’s phone is mostly on, but there is a small chance it’s off (**20% chance**).
- **Average Daily Activations**: On average, each user has **76 screen-on activations per day**.
- **Time Bins**: The 24-hour period starts at **16:00** and ends at **15:59** the next day, and the day is divided into **15-minute bins** for a total of **96 bins per day per user**.

### Event Count Distribution
- **Events per Bin**: Each 15-minute bin contains an integer value representing the number of events (e.g., screen-on or app launch) that occurred during that interval.
- **Daily Distribution**: Events are distributed across all time bins, with most events occurring during daytime hours (**07:00 to 23:00**), and fewer events occurring during nighttime hours (**23:00 to 07:00**).
- **Poisson Distribution**: The total number of events per day is Poisson-distributed around an average of **76 events**.
- **Day vs. Night Allocation**: About **85% of the events** are allocated to daytime bins, while **15%** are assigned to nighttime bins, matching typical user behavior.

## Code for Synthetic Data Generation

The synthetic data is generated using Python. Below is the code that generates the user data for the SensibleSleep project.

### File Location
- The code for generating synthetic data is located in the `sensible_sleep/synthetic_data.py` file.

### Code Explanation

#### `generate_user_data`
This function generates synthetic data for a single user, providing screen-on events for each 15-minute bin across multiple days.

#### `generate_synthetic_data`
This function generates synthetic data for multiple users, combining the results into a single DataFrame.
