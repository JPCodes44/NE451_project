# Justin Mak
# Assignment 3 top 20 spotify artists
# 2025/11/22
# Generates line plots comparing the total execution time (total_time_s) for different dictionary (map) implementations,
# using timing results from CSV files. The script reads timing data for various dictionary types, reshapes it for plotting,
# and visualizes the performance of each implementation as the dataset size varies.

# Input:  Timing results from the results/topk folder (CSV files)
# Output: Line plots comparing the total_time_s for each dictionary implementation by strategy


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Ensure script runs relative to its own directory


import pandas as pd
import plotly.express as px


def plot_avg_total_time_plotly(csv_path):
    df = pd.read_csv(csv_path)
    avg_times = df.groupby('strategy')['total_time_s'].mean().reset_index()
    fig = px.bar(
        avg_times,
        x='strategy',
        y='total_time_s',
        title='Average Total Time by Strategy (Averaged Across All n)',
        labels={'total_time_s': 'Average Total Time (s)', 'strategy': 'Strategy'},
        text='total_time_s',
    )
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(
        xaxis_tickangle=-20,
        yaxis_title='Average Total Time (s)',
        xaxis_title='Strategy',
        title_font=dict(size=28),
        xaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
        yaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
        legend_font=dict(size=18)
    )
    fig.show()

# Example usage:
plot_avg_total_time_plotly('results/topk/topk-builtin_dict.csv')


