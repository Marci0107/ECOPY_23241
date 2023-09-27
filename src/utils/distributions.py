import random


class FirstClass():

class SecondClass():
    def __init__(self, rand):
        self.random = random.random()

class UniformDistribution():
    def __init__(self, rand, a, b):
        self.lower_bound = a
        self.upper_bound = b
        self.rand_gen = rand
