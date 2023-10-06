import random
import math
import numpy as np


class LaplaceDistribution():
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        prefactor = 1 / (2 * self.scale)
        exponent = -(abs(x - self.loc) / self.scale)
        probability_density = prefactor * math.exp(exponent)
        return probability_density

    def cdf(self, x):
        if x < self.loc:
            cumulative_probability = 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            cumulative_probability = 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)

        return cumulative_probability

    def ppf(self, p):
        if p < 0.5:
            x = self.loc + self.scale * math.log(2 * p)
        else:
            x = self.loc - self.scale * math.log(2 - 2 * p)
        return x

    def gen_rand(self):
        u = random.random()
        if u < 0.5:
            x = self.loc + self.scale * math.log(2 * u)
        else:
            x = self.loc - self.scale * math.log(2 - 2 * u)

        return x

    def mean(self):
        expected_value = self.loc
        return expected_value

    def variance(self):
        return 2 * self.scale ** 2

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 3

    def mvsk(self):
        return [self.loc, 2 * self.scale ** 2, 0, 3]


class ParetoDistribution():
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x >= self.scale:
            probability_density = (self.shape * self.scale ** self.shape) / (x ** (self.shape + 1))
        else:
            probability_density = 0

        return probability_density

    def cdf(self, x):
        if x >= self.scale:
            cumulative_probability = 1 - (self.scale / x) ** self.shape
        else:
            cumulative_probability = 0

        return cumulative_probability

    def ppf(self, p):
        if 0 <= p < 1:
            quantile = self.scale / (1 - p) ** (1 / self.shape)
        else:
            raise ValueError("Az inverz kumulatív eloszlásfüggvény csak 0 és 1 közötti valószámokra értelmezett.")

        return quantile

    def gen_rand(self):
        u = self.rand.random()
        random_numbers = self.scale / (1 - u) ** (1 / self.shape)
        return random_numbers

    def mean(self):
        if self.shape > 1:
            expected_value = (self.shape * self.scale) / (self.shape - 1)
            return expected_value
        else:
            float('inf')

    def variance(self):
        if self.shape > 2:
            variance = (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))
            return variance
        else:
            float('inf')

    def skewness(self):
        if self.shape > 3:
            skewness = ((2 * (1 + self.shape)) / (self.shape - 3)) * (((self.shape - 2) / self.shape) ** 0.5)
            return skewness
        else:
            float('inf')

    def ex_kurtosis(self):
        if self.shape > 4:
            ex_kurtosis = (6 * (self.shape ** 3 + self.shape ** 2 - 6 * self.shape - 2)) / (self.shape * (self.shape - 3) * (self.shape - 4))
            return ex_kurtosis
        else:
            float('inf')

    def mvsk(self):
        expected_value = (self.shape * self.scale) / (self.shape - 1)
        variance = (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))
        skewness = ((2 * (1 + self.shape)) / (self.shape - 3)) * (((self.shape - 2) / self.shape) ** 0.5)
        ex_kurtosis = (6 * (self.shape ** 3 + self.shape ** 2 - 6 * self.shape - 2)) / (self.shape * (self.shape - 3) * (self.shape - 4))
        return [expected_value, variance, skewness, ex_kurtosis]