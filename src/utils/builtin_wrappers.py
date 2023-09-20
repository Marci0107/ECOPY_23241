# 7., Írjon egy függvényt ami megvizsgálja, hogy a listában létezik-e egy adott elem
# függvény név: contains_value
# bemeneti paraméterek: input_list, element
# kimeneti típus: bool
def contains_value(input_list, element):
    if element in input_list:
        print("True")
    else:
        print("False")
# 8., Írjon egy függvényt ami megvizsgálja, hogy hány elem található a listában.
# függvény név: number_of_elements_in_list
# bemeneti paraméterek: input_list
# kimeneti típus: int
def number_of_elements_in_list(input_list):
    return len(input_list)
# 9., Írjon egy függvényt ami törli az összes elemet a listából
# függvény név: remove_every_element_from_list
# bemeneti paraméterek: input_list
# kimeneti típus: None
def remove_every_element_from_list(input_list):
    input_list.clear()
    return input_list
# 10., Írjon egy függvényt ami fordított sorrendben adja vissza a lista összes elemét
# függvény név: reverse_list
# bemeneti paraméterek: input_list
# kimeneti típus: List
def reverse_list(input_list):
    new_list = input_list[::-1]
    return new_list
# 11., Írjon egy függvényt ami vissza adja a bemeneti lista páratlan elemeit
# függvény név: odds_from_list
# bemeneti paraméterek: input_list
# kimeneti típus: List
def odds_from_list(input_list):

    return [x for x in input_list if x % 2 == 1]
# 12., Írjon egy függvényt ami megszámolja a bemeneti lista páratlan elemeit
# függvény név: number_of_odds_in_list
# bemeneti paraméterek: input_list
# kimeneti típus: int
def number_of_odds_in_list(input_list):

    return len([x for x in input_list if x % 2 == 1])
# 13., Írjon egy függvényt ami visszaadja, hogy a bemeneti listának van-e páratlan eleme
# függvény név: contains_odd
# bemeneti paraméterek: input_list
# kimeneti típus: bool
def contains_odd(input_list):
    count = 0
    for i in range(len(input_list)):
        if input_list[i] % 2 == 1:
            count = 1
    if count == 0:
        return False
    else:
        return True
# 14., Írjon egy függvényt ami visszaadja a 2. legnagyobb elemet listában
# függvény név: second_largest_in_list
# bemeneti paraméterek: input_list
# kimeneti típus: int
def second_largest_in_list(input_list):
    max_element_no = input_list.index(max(input_list))
    input_list.remove(input_list[max_element_no])
    return max(input_list)
# 15., Írjon egy függvényt ami kiszámítja a lista elemek összegét
# függvény név: sum_of_elements_in_list
# bemeneti paraméterek: input_list
# kimeneti típus: float
def sum_of_elements_in_list(input_list):
    sum_of_elements = 0
    for i in range(len(input_list)):
        sum_of_elements += input_list[i]
    return float(sum_of_elements)

# 1. Hozz létre egy dictionary-t (dict.) amelyhez a két kulcs 'a' és 'b' és a hozzájuk tartozó értékek rendre 9 és [12, 'c']
dict = {
    'a': 9,
    'b': [12, 'c']
}
# 2. Kérd le az előző dictionary 'a' kulcsán lévő értéket
print(dict['a'])
# 3. Kérd le az előző dictionary 'd' kulcsán lévő értéket olyan módon, hogy ne hibát kapj, hanem Null legyen a visszatérési érték
for key, value in dict.items():  #accessing values
    print(value,end='d')
# 4., Írjon egy függvényt amely töröl egy megadott kulcsú elemet a dict.-ből
# függvény név: remove_key
# bemeneti paraméterek: input_dict, key
# kimeneti típus: Dict
def remove_key(input_dict, key):
    if key in input_dict.keys():
        del input_dict[key]
        return input_dict
    else:
        return input_dict
# 5., Írjon egy függvényt amely a sorba rendezi a kulcs-érték párokat a kulcs értéke szerint egy dictionary-ben
# függvény név: sort_by_key
# bemeneti paraméterek: input_dict
# kimeneti típus: Dict
def sort_by_key(input_dict):
    return dict(sorted(input_dict.items()))
# 6., Írjon egy függvényt amely összeadja a dict.-ben található összes értéket
# függvény név: sum_in_dict
# bemeneti paraméterek: input_dict
# kimeneti típus: float
def sum_in_dict(input_dict):
    sum = 0
    for value in input_dict.values():
        sum += value
    return sum
# 7., Írjon egy függvényt amely összekapcsol 2 dictionary-t 1 dict.-ben
# függvény név: merge_two_dicts
# bemeneti paraméterek: input_dict1, input_dict2
# kimeneti típus: Dict
def merge_two_dicts(input_dict1, input_dict2):
    merged_dict = input_dict1.copy()
    merged_dict.update(input_dict2)
    return merged_dict
# 8., Írjon egy függvényt amely összekapcsol n dictionary-t 1 dict.-ben
# függvény név: merge_dicts
# bemeneti paraméterek: *dicts
# kimeneti típus: Dict
def merge_dicts(*dicts):
    merged_dict = {}
    for dictionary in dicts:
        merged_dict.update(dictionary)
    return merged_dict
Írjon egy függvényt amely a bemeneti, pozitív egész számokat tartalmazó listát kiválogatja páros és páratlan számokra, és visszaad egy olyan dictionary-t, amelyben a kulcs az 'even' és 'odd', az értékek, pedig a listák.
# függvény név: sort_list_by_parity
# bemeneti paraméterek: input_list
# kimeneti típus: Dict
def sort_list_by_parity(input_list):
    even_numbers = list()
    odd_numbers = list()
    for i in range(len(input_list)):
        if input_list[i] % 2 == 0:
             even_numbers.append(i)
        else:
                odd_numbers.append(i)
return {'even': even_numbers, 'odd': odd_numbers}
# 10., Írjon egy függvényt amely a bemenetként kapott dictionary értékeinél található listák átlagait adja vissza egy új dictionary-be. {"some_key":[1,2,3,4],"another_key":[1,2,3,4],....} -> {"some_key":mean_of_values,"another_key":mean_of_values,....}
# függvény név: mean_by_key_value
# bemeneti paraméterek: input_dict
# kimeneti típus: Dict
def mean_by_key_value(input_dict):
        result = {}
        for key, value in input_dict.items():