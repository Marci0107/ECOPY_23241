
def contains_value(input_list, element):
    if element in input_list:
        print("True")
    else:
        print("False")

def number_of_elements_in_list(input_list):
    return len(input_list)

def remove_every_element_from_list(input_list):
    input_list.clear()
    return input_list

def reverse_list(input_list):
    new_list = input_list[::-1]
    return new_list

def odds_from_list(input_list):

    return [x for x in input_list if x % 2 == 1]

def number_of_odds_in_list(input_list):

    return len([x for x in input_list if x % 2 == 1])

def contains_odd(input_list):
    count = 0
    for i in range(len(input_list)):
        if input_list[i] % 2 == 1:
            count = 1
    if count == 0:
        return False
    else:
        return True

def second_largest_in_list(input_list):
    max_element_no = input_list.index(max(input_list))
    input_list.remove(input_list[max_element_no])
    return max(input_list)

def sum_of_elements_in_list(input_list):
    sum_of_elements = 0
    for i in range(len(input_list)):
        sum_of_elements += input_list[i]
    return float(sum_of_elements)

dict = {
    'a': 9,
    'b': [12, 'c']
}

print(dict['a'])

for key, value in dict.items():  #accessing values
    print(value,end='d')

def remove_key(input_dict, key):
    if key in input_dict.keys():
        del input_dict[key]
        return input_dict
    else:
        return input_dict

def sort_by_key(input_dict):
    return dict(sorted(input_dict.items()))

def sum_in_dict(input_dict):
    sum = 0
    for value in input_dict.values():
        sum += value
    return sum

def merge_two_dicts(input_dict1, input_dict2):
    merged_dict = input_dict1.copy()
    merged_dict.update(input_dict2)
    return merged_dict

def merge_dicts(*dicts):
    merged_dict = {}
    for dictionary in dicts:
        merged_dict.update(dictionary)
    return merged_dict

def sort_list_by_parity(input_list):
    even_numbers = list()
    odd_numbers = list()
    for i in range(len(input_list)):
        if input_list[i] % 2 == 0:
             even_numbers.append(i)
        else:
                odd_numbers.append(i)
    return {'even': even_numbers, 'odd': odd_numbers}
def mean_by_key_value(input_dict):
    dummydict = dict()
    for i in input_dict:
        if len(input_dict[i]) < 2:
            dummydict[i] = input_dict[i][0]
        else:
            dummydict[i] = sum(input_dict[i]) / len(input_dict[i])

    return dummydict

def count_frequency(input_list):
    # Initialize an empty dictionary to store the frequencies
    frequency_dict = {}

    # Iterate through the elements in the input list
    for item in input_list:
        # If the item is not in the dictionary, add it with a count of 1
        if item not in frequency_dict:
            frequency_dict[item] = 1
        else:
            # If the item is already in the dictionary, increment its count
            frequency_dict[item] += 1

    return frequency_dict