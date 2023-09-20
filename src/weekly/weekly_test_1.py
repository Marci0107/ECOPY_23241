def evens_from_list(input_list):
    return [x for x in input_list if x % 2 == 0]

def every_element_is_odd(input_list):
    count = 0
    for i in input_list:
        if i % 2 == 0:
            count += 1
    if count == 0:
        return True
    else:
        return False

def kth_largest_in_list(input_list, kth_largest):
    input_list.sort(reverse=True)
    return input_list[kth_largest - 1]

def cumavg_list(input_list):
    rolling_averages = []

    for i in range(len(input_list)):
        start_index = 0
        end_index = i + 1
        window_values = input_list[start_index:end_index]
        average = sum(window_values) / len(window_values)
        rolling_averages.append(average)

    return rolling_averages

def element_wise_multiplication(input_list1, input_list2):
    import numpy as np
    arr_input_list1 = np.array(input_list1)
    arr_input_list2 = np.array(input_list2)

    return list(arr_input_list1 * arr_input_list2)

def merge_lists(*lists):
    merged_list = []
    for list in lists:
        merged_list.extend(list)
    return merged_list

def squared_odds(input_list):
    return [x**2 for x in input_list if x % 2 != 0]

def reverse_sort_by_key(input_dict):
    return dict(sorted(input_dict.items(), key=lambda x: x[0], reverse=True))

def sort_list_by_divisibility(input_list):
    by2_numbers = []
    by5_numbers = []
    by10_numbers = []
    other_numbers = []

    for i in input_list:
        if i % 2 == 0 and i % 5 == 0:
            by10_numbers.append(i)
        elif i % 2 == 0:
            by2_numbers.append(i)
        elif i % 5 == 0:
            by5_numbers.append(i)
        else:
            other_numbers.append(i)

    result_dict = {
        'by_two': by2_numbers,
        'by_five': by5_numbers,
        'by_two_and_five': by10_numbers,
        'by_none': other_numbers
    }

    return result_dict