# 1., Hozzon létre egy listát a következő elemekkel a Subscript operátor segítségével: 17, 18, 3.14, a, alma
x = [17, 18, 3.14, "a", "alma"]
# 2., Hozzon létre egy listát a list() függvény alkalmazásával: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
list1 = list(tuple)
print(list1)
# 3., Adja vissza a lista 1. indexén lévő elemét
x[1]
# 4., Adja vissza a lista 1 elemét
list1[0]
# 5., Adja vissza a lista legnagyobb elemét
max(list1[:])
# 6. Adja vissza a lista legnagyobb elemének az indexét
list1.index(max(list1[:]))
# 7., Írjon egy függvényt ami megvizsgálja, hogy a listában létezik-e egy adott elem
# függvény név: contains_value
# bemeneti paraméterek: input_list, element
# kimeneti típus: bool
def contains_value(input_list, element):
    if element in input_list:
        print(True)
    else:
        print(False)
    return
contains_value(list1, 11)
# 8., Írjon egy függvényt ami megvizsgálja, hogy hány elem található a listában.
# függvény név: number_of_elements_in_list
# bemeneti paraméterek: input_list
# kimeneti típus: int
def number_of_elements_in_list(input_list):
    print(len(input_list))
number_of_elements_in_list(list1)
# 9., Írjon egy függvényt ami törli az összes elemet a listából
# függvény név: remove_every_element_from_list
# bemeneti paraméterek: input_list
# kimeneti típus: None