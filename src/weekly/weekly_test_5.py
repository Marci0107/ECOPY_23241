import pandas as pd
import random

food = pd.read_csv("C:/Users/User\Documents\GitHub\ECOPY_23241\data\chipotle.tsv", delimiter='\t')

def change_price_to_float(input_df):
    food = input_df.copy()
    food['item_price'] = food['item_price'].str.replace('$', '').astype(float)
    return food

def number_of_observations(input_df):
    new_df = input_df.copy()
    return new_df.shape[0]

def items_and_prices(input_df):
    new_df = input_df.copy()
    selected_columns = new_df[["item_name", "item_price"]]
    return selected_columns

def sorted_by_price(input_df):
    new_df = input_df.copy()
    sorted_df = new_df.sort_values(by="item_price", ascending=False)
    return sorted_df

def avg_price(input_df):
    new_df = input_df.copy()
    average_price = new_df["item_price"].mean()
    return average_price

def unique_items_over_ten_dollars(input_df):
    new_df = input_df.copy()
    filtered_df = new_df[new_df["item_price"] > 10]
    # Duplikációk eltávolítása
    unique_items_df = filtered_df.drop_duplicates(subset=["item_name", "choice_description", "item_price"])
    return unique_items_df[["item_name", "choice_description", "item_price"]]

def items_starting_with_s(input_df):
    new_df = input_df.copy()
    selected_items = new_df[new_df['item_name'].str.startswith('S')]
    unique_selected_items_df = selected_items.drop_duplicates(subset=["item_name"])
    return unique_selected_items_df["item_name"].reset_index(drop=True)

def first_three_columns(input_df):
    new_df = input_df.copy()
    new_df = new_df.iloc[:, 0:3]
    return new_df

def every_column_except_last_two(input_df):
    new_df = input_df.copy()
    remaining_columns = new_df.iloc[:, :-2]
    return remaining_columns

def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    new_df = input_df.copy()
    filtered_df = new_df[input_df[column_to_filter].isin(rows_to_keep)]
    selected_columns_df = filtered_df[columns_to_keep]
    return selected_columns_df

def generate_quartile(input_df):
    new_df = input_df.copy()
    new_df['Quartile'] = pd.cut(input_df['item_price'], [-1, 10, 20, 30, float('inf')], labels=['low-cost', 'medium-cost', 'high-cost', 'premium'], right=False).astype('object')
    return new_df

def average_price_in_quartiles(input_df):
    new_df = input_df.copy()
    avg_price_df = new_df.groupby('Quartile')['item_price'].mean().reset_index(drop=True)
    return avg_price_df

def minmaxmean_price_in_quartile(input_df):
    new_df = input_df.copy()
    result = new_df.groupby('Quartile')['item_price'].agg(['min', 'max', 'mean']).reset_index(drop=True)
    result.columns = ['min', 'max', 'mean']
    return result

def gen_uniform_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    for _ in range(number_of_trajectories):
        random.seed(42)
        trajectories = []

        for _ in range(number_of_trajectories):
            trajectory = []
            cumulative_sum = 0.0

            for _ in range(length_of_trajectory):
                random_value = distribution.gen_random()
                cumulative_sum += random_value
                cumulative_average = cumulative_sum / (len(trajectory) + 1)  # Számoljuk a kumulatív átlagot
                trajectory.append(cumulative_average)  # Hozzáadjuk a kumulatív átlagot a belső listához

            trajectories.append(trajectory)  # Hozzáadjuk a belső listát az eredmény listához

        return trajectories

def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    for _ in range(number_of_trajectories):
        random.seed(42)
        trajectories = []

        for _ in range(number_of_trajectories):
            trajectory = []
            cumulative_sum = 0.0

            for _ in range(length_of_trajectory):
                random_value = distribution.gen_rand()
                cumulative_sum += random_value
                cumulative_average = cumulative_sum / (len(trajectory) + 1)  # Számoljuk a kumulatív átlagot
                trajectory.append(cumulative_average)  # Hozzáadjuk a kumulatív átlagot a belső listához

            trajectories.append(trajectory)  # Hozzáadjuk a belső listát az eredmény listához

        return trajectories

def gen_laplace_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    for _ in range(number_of_trajectories):
        random.seed(42)
        trajectories = []

        for _ in range(number_of_trajectories):
            trajectory = []
            cumulative_sum = 0.0

            for _ in range(length_of_trajectory):
                random_value = distribution.gen_rand()
                cumulative_sum += random_value
                cumulative_average = cumulative_sum / (len(trajectory) + 1)  # Számoljuk a kumulatív átlagot
                trajectory.append(cumulative_average)  # Hozzáadjuk a kumulatív átlagot a belső listához

            trajectories.append(trajectory)  # Hozzáadjuk a belső listát az eredmény listához

        return trajectories

def gen_cauchy_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    for _ in range(number_of_trajectories):
        random.seed(42)
        trajectories = []

        for _ in range(number_of_trajectories):
            trajectory = []
            cumulative_sum = 0.0

            for _ in range(length_of_trajectory):
                random_value = distribution.gen_random()
                cumulative_sum += random_value
                cumulative_average = cumulative_sum / (len(trajectory) + 1)  # Számoljuk a kumulatív átlagot
                trajectory.append(cumulative_average)  # Hozzáadjuk a kumulatív átlagot a belső listához

            trajectories.append(trajectory)  # Hozzáadjuk a belső listát az eredmény listához

        return trajectories

def gen_chi2_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    for _ in range(number_of_trajectories):
        random.seed(42)
        trajectories = []

        for _ in range(number_of_trajectories):
            trajectory = []
            cumulative_sum = 0.0

            for _ in range(length_of_trajectory):
                random_value = distribution.gen_rand()
                cumulative_sum += random_value
                cumulative_average = cumulative_sum / (len(trajectory) + 1)  # Számoljuk a kumulatív átlagot
                trajectory.append(cumulative_average)  # Hozzáadjuk a kumulatív átlagot a belső listához

            trajectories.append(trajectory)  # Hozzáadjuk a belső listát az eredmény listához

        return trajectories