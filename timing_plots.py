# Justin Mak
# Assignment 3 top 20 spotify artists
# 2025/11/22
# Generates grouped bar chart plots comparing the average lookup and add times for different dictionary (map) implementations,
# using timing results from CSV files. The script reads timing data for various dictionary types, reshapes it for plotting,
# and visualizes the performance of each implementation as the dataset size varies.

# Input:  Timing results from the results/timings folder (CSV files)
# Output: Bar chart plots comparing the lookup time average and add time average for each dictionary implementation

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Ensure script runs from its own directory

import pandas as pd
import plotly.express as px
import glob


def load_and_prepare(csv_path: str, ignore_cols: set) -> pd.DataFrame | None:
    """
    Read a timing CSV file, filter for average timing metrics, and reshape to long format.

    Args:
        csv_path (str): Path to the CSV file.
        ignore_cols (set): Columns to ignore when selecting metrics.

    Returns:
        pd.DataFrame | None: Long-form DataFrame with columns ['n', 'metric', 'value', 'source'],
        or None if no relevant metrics are found.
    """
    df = pd.read_csv(csv_path)

    if "n" not in df.columns:
        raise ValueError(f"'n' column not found in {csv_path}")

    # Find numeric columns that are not in ignore_cols
    numeric_cols = [
        col for col in df.columns
        if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Only keep the average timing columns we care about
    desired_metrics = {"lookup_time_avg_ns", "add_time_avg_ns"}
    numeric_cols = [col for col in numeric_cols if col in desired_metrics]

    if not numeric_cols:
        print(f"No average timing columns found in {csv_path}, skipping.")
        return None

    # Reshape DataFrame to long format for plotting
    df_long = df[["n"] + numeric_cols].melt(
        id_vars="n",
        value_vars=numeric_cols,
        var_name="metric",
        value_name="value",
    )

    # Add a column indicating the source dictionary (from filename)
    df_long["source"] = os.path.splitext(os.path.basename(csv_path))[0]
    return df_long



def main():
    '''
    Main entry point for generating grouped bar charts of timing metrics from CSV files.

    Inputs:
        None. The function reads all CSV files in the 'results/timings/' directory.

    Outputs:
        None. The function displays interactive grouped bar charts for each specified timing metric.

    Description:
        This function aggregates timing data from multiple CSV files, reshapes the data for plotting,
        and generates grouped bar charts for each timing metric (average and total lookup/add times).
        The charts compare different strategies across varying numbers of artists, with enhanced legend readability.
    '''

    # Combine all timing CSVs if you want all strategies together
    csv_files = glob.glob("results/timings/*.csv")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Select the metrics you want to plot
    metrics = [
        "lookup_time_avg_ns",
        "add_time_avg_ns",
        "lookup_time_total_ns",
        "add_time_total_ns"
    ]

    # Melt the DataFrame to long format for easier plotting
    df_long = df.melt(
        id_vars=["n", "strategy"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Value"
    )

    # Plot grouped bar chart for each metric
    for metric in metrics:
        fig = px.bar(
            df_long[df_long["Metric"] == metric],
            x="n",
            y="Value",
            color="strategy",
            barmode="group",
            title=f"{metric} grouped by strategy",
            labels={"n": "n (number of artists)", "Value": metric, "strategy": "Strategy"}
        )
        # Make legend font bigger
        fig.update_layout(
            legend=dict(
                font=dict(
                    size=18  # Adjust this value as needed
                )
            )
        )
        fig.show()





if __name__ == "__main__":
    main()
