import pandas as pd
import matplotlib.pyplot as plt
import random
euro12 = pd.read_csv("C:/Users/User\Documents\GitHub\ECOPY_23241\data\Euro_2012_stats_TEAM.csv")

def number_of_participants(input_df):
    new_df = input_df.copy()
    return new_df.shape[0]

def goals(input_df):
    new_df = input_df.copy()
    goals_df = new_df[['Team', 'Goals']]
    return goals_df

def sorted_by_goal(input_df):
    new_df = input_df.copy()
    sorted_df = new_df.sort_values(by='Goals', ascending=False)
    return sorted_df

def avg_goal(input_df):
    new_df = input_df.copy()
    average_goal = new_df['Goals'].mean()
    return average_goal

def countries_over_five(input_df):
    new_df = input_df.copy()
    over_five = new_df[new_df['Goals'] >= 6]
    return over_five

def countries_starting_with_g(input_df):
    new_df = input_df.copy()
    selected_countries = new_df[new_df['Team'].str.startswith('G')]
    return selected_countries

def first_seven_columns(input_df):
    new_df = input_df.copy()
    new_df = new_df.iloc[:, 0:7]
    return new_df

def every_column_except_last_three(input_df):
    new_df = input_df.copy()
    remaining_columns = new_df.iloc[:, :-3]
    return remaining_columns

def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    new_df = input_df.copy()
    # Az oszlopok kiválasztása a "columns_to_keep" listából
    selected_columns = new_df[columns_to_keep]

    # Sorok szűrése a "column_to_filter" oszlop alapján
    filtered_rows = new_df[input_df[column_to_filter].isin(rows_to_keep)]

    # Visszatérés a kiválasztott oszlopok és szűrt sorok tartalmával
    return selected_columns.merge(filtered_rows, on=columns_to_keep)

def generate_quartile(input_df):
    new_df = input_df.copy()
    # Quartile értékek meghatározása a "Goals" oszlop alapján
    new_df['Quartile'] = pd.qcut(input_df['Goals'], q=[0, 0.25, 0.5, 0.75, 1], labels=[4, 3, 2, 1])
    return new_df

def average_yellow_in_quartiles(input_df):
    new_df = input_df.copy()
    new_df['Quartile'] = pd.qcut(input_df['Goals'], q=[0, 0.25, 0.5, 0.75, 1], labels=[4, 3, 2, 1])
    new_df['Average passes'] = new_df.groupby('Quartile')['Passes'].mean()
    return new_df

def minmax_block_in_quartile(input_df):
    new_df = input_df.copy()
    new_df['Quartile'] = pd.qcut(input_df['Goals'], q=[0, 0.25, 0.5, 0.75, 1], labels=[4, 3, 2, 1])
    new_df['Minimum blocks'] = new_df.groupby('Quartile')['Blocks'].min()
    new_df['Maximum blocks'] = new_df.groupby('Quartile')['Blocks'].max()
    return new_df

def scatter_goals_shots(input_df):
    new_df = input_df.copy()
    fig, ax = plt.subplots()
    ax.scatter(new_df['Goals'], new_df['Shots on target'])

    ax.set_title('Goals and Shot on target')
    ax.set_xlabel('Goals')
    ax.set_ylabel('Shots on target')

    plt.show()
    return fig

