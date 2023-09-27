import random
import math
import pyerf

class UniformDistribution():
    def __init__(self, rand, a, b):
        self.rand = rand
        self.a = a
        self.b = b
    def pdf(self, x):
        self.x = x
        if self.a <= self.x <= self.b:
            self.x = 1 / (self.b - self.a)
        else:
            self.x = 0
        return self.x
    def cdf(self, x):
        self.x = x
        if self.a <= self.x <= self.b:
            self.x = (self.x - self.a) / (self.b - self.a)
        elif self.x < self.a:
                self.x = 0
        else:
                self.x = 1
        return self.x

    def ppf(self, p):
        self.p = p
        if 0 <= self.p <= 1:
            self.p = self.a + (self.b-self.a)*self.p
        return self.p

    def gen_random(self):
        return random.uniform(self.a, self.b)

    def mean(self):
        return 0.5*(self.a+self.b)

    def median(self):
        return 0.5*(self.a+self.b)

    def variance(self):
        return 1/12*(self.b-self.a)^2

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return -6/5

    def mvsk(self):
        return [0.5*(self.a+self.b), 1/12*(self.b-self.a)*(self.b-self.a), 0, -6/5]

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale


    def pdf(self, x):
        self.x = x
        sd = (self.scale) ** 0.5
        denominator = sd*(2*math.pi)**0.5
        exponent = (-0.5)*(((self.x-self.loc)/sd)**2)
        return 1/denominator*(math.e)**exponent

    def cdf(self, x):
        sd = (self.scale)**0.5
        denominator = sd*2**0.5
        self.x = x
        return 0.5*(1 + math.erf((self.x-self.loc)/denominator))

    def ppf(self, p):
        sd = (self.scale) ** 0.5
        denominator = sd * 2 ** 0.5
        self.p = p
        return pyerf.erfinv(2*p-1)*denominator+self.loc

    def gen_random(self):
        return random.normalvariate(self.loc, self.scale)

    def mean(self):
        return self.loc

    def median(self):
        return self.loc

    def variance(self):
        return self.scale

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 0

    def mvsk(self):
        return [self.loc, self.scale, 0, 0]

class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        self.x = x
        den_parentheses = (self.x-self.loc)/self.scale
        denominator = self.scale*math.pi*(1+den_parentheses**2)
        return 1/denominator

    def cdf(self, x):
        self.x = x
        parentheses = (self.x-self.loc)/self.scale

        return 1/math.pi*math.atan(parentheses) + 0.5

    def ppf(self, p):
        self.p = p
        parentheses = math.pi*(self.p - 0.5)
        return self.loc + self.scale*math.tan(parentheses)

    def gen_random(self):
        return random.standard_cauchy()*self.scale + self.loc

    def mean(self):
        raise Exception("Moments undefined")

    def median(self):
        raise Exception("Moments undefined")

    def skewness(self):
        raise Exception("Moments undefined")

    def ex_kurtosis(self):
        raise Exception("Moments undefined")

    def mvsk(self):
        return [self.mean(), self.median(), self.skewness(), self.ex_kurtosis()]