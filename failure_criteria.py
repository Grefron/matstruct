import math

from materials import quadratic


class VonMises:
    pass


class TsaiHill:
    def reserve_factor(self, load):
        a = ((sigma1 / s1) ** 2 - (sigma1 * sigma2) / s1 ** 2 +
             (sigma2 / s2) ** 2 + (sigma6 / s6) ** 2)
        RF = math.sqrt(1 / a)
        print('RF', RF)
        return RF


class Norris:
    def reserve_factor(self, load):
        return ((sigma1 / s1) ** 2 + (sigma2 / s2) ** 2 + (sigma6 / s6) ** 2)


class TsaiWu:
    def reserve_factor(self, load):
        F1 = 1 / s1t - 1 / s1c
        F2 = 1 / s2t - 1 / s2c
        F11 = 1 / (s1t * s1c)
        F22 = 1 / (s2t * s2c)
        F66 = 1 / (s6 ** 2)
        F12 = - 0.5 * math.sqrt(F11 * F22)  # estimate

        # p425 reserve factor
        a = F11 * sigma1 ** 2 + F22 * sigma2 ** 2 + F66 * sigma6 ** 2 + F12 * sigma1 * sigma2
        b = F1 * sigma1 + F2 * sigma2 + F66 * sigma6
        RF, _ = quadratic(a, b, -1)
        print('RF!', RF)

        return (F1 * sigma1 + F2 * sigma2 +
                F11 * sigma1 ** 2 + F22 * sigma2 ** 2 +
                2 * F12 * sigma1 * sigma2 + F66 * sigma6 ** 2)


class PuckSchneider:
    def reserve_factor():
        pass


class Hashin:
    def reserve_factor():
        sigma3 = 0
        s23 = 1e9
        sigma23 = 0
        sigma13 = 1e6
        return ((1 / s2t ** 2) * (sigma2 + sigma3) ** 2 + (1 / s23 ** 2) * (sigma23 ** 2 - sigma2 * sigma3) +
                (1 / s6 ** 2) * (sigma6 ** 2 + sigma13 ** 3))


class HashinRotem:
    def reserve_factor():
        pass


class Christensen:
    def reserve_factor():
        pass