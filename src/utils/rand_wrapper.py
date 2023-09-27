import random

random.random()

random.randint(1, 100)

random.seed(42)
random.randint(1, 100)

def random_from_list(input_list):
    len_input_list = len(input_list)
    random_no = random.randint(0, len_input_list - 1)
    return input_list[random_no]

def random_sublist_from_list(input_list, number_of_elements):
    return random.choices(input_list, k = number_of_elements)

def random_sublist_from_list2(input_list, number_of_elements):
    random_numbers_list = random.sample(input_list, number_of_elements)
    return random_numbers_list

def random_from_string(input_string):
    random_character = random.choice(input_string)
    return random_character

def hundred_small_random():
    hundred_random_numbers = []
    for i in range(100):
        random_zero_to_one = random.random()
        hundred_random_numbers.append(random_zero_to_one)
    return hundred_random_numbers

def hundred_large_random():
    hundred_random_int_numbers = []
    for i in range(100):
        random_ten_to_thousand = random.randint(10, 1000)
        hundred_random_int_numbers.append(random_ten_to_thousand)
    return hundred_random_int_numbers

def five_random_number_div_three():
    list_1 = [value for value in list(range(9, 1001)) if value % 3 == 0]
    return random.sample(list_1, k = 5)

def five_random_number_div_three2():
    five_random_numbers = []
    while len(five_random_numbers) < 5:
        rand_between_nine_thousand = random.randint(9, 1000)
        if rand_between_nine_thousand % 3 == 0:
            five_random_numbers.append(rand_between_nine_thousand)
    return five_random_numbers

def random_reorder(input_list):
    return random.sample(input_list, k = len(input_list))

def random_reorder2(input_list):
    shuffled_list = input_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list


def uniform_one_to_five():
    random_uniform_num = random.uniform(1, 6)
    return random_uniform_num
