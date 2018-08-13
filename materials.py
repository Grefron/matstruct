import math
from itertools import product
from collections import namedtuple
import sqlite3
from matplotlib import pyplot as plt


import numpy as np


# http://wstein.org/edu/2010/480b/projects/05-lamination_theory/A%20summary%20of%20Classical%20Lamination%20Theory.pdf
# from .failure_criteria import TsaiHill


class Micromechanics:
    def __init__(self, ply):
        self.ply = ply

    def get(self, names):
        attrs = []
        for name in names.replace(', ', ' ').split():
            try:
                attrs.append(getattr(self.ply, name))
            except AttributeError:
                attrs.append(getattr(self, name))
        return tuple(attrs)


class RuleOfMixtures(Micromechanics):
    def __init__(self, ply, *args, **kwargs):
        super().__init__(ply, *args, **kwargs)
        self.filament_misalignment_factor = kwargs.get("filament_misalignment_factor", 0.95)

    @property
    def a(self):
        return self.filament_misalignment_factor

    @property
    def E_1(self):
        a, v_f, v_m, E_f, E_m = self.get('a v_f v_m E_f E_m')
        return a * (v_f * E_f + E_m * v_m)

    @property
    def E_2(self):
        v_f, v_m, E_f, E_m = self.get('v_f v_m E_f E_m')
        return 1 / (v_f / E_f + v_m / E_m)

    @property
    def G_12(self):
        #  Kollar p442
        v_f, v_m, G_f, G_m = self.get('v_f v_m G_f G_m')
        return 1 / (v_f / G_f + v_m / G_m)

    @property
    def G_23(self):
        #  Kollar p442
        v_f, v_m, G_f, G_m = self.get('v_f v_m G_f G_m')
        return 1 / (v_f / G_f + v_m / G_m)

    @property
    def nu_12(self):
        #  Kollar p442
        v_f, v_m, nu_f, nu_m = self.get('v_f v_m nu_f nu_m')
        return v_f * nu_f + v_m * nu_m

    @property
    def nu_23(self):
        #  Kollar p442
        return self.E_2 / (2 * self.G_23) - 1


class Tsai(RuleOfMixtures):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contiguity_factor = kwargs.get("contiguity_factor", 0.25)

    def tsai_constant(self, material):
        return material.modulus / (2 * (1 - material.poissonratio))

    @property
    def C(self):
        return self.contiguity_factor

    @property
    def C_f(self):
        return self.tsai_constant(self.ply.fibre)

    @property
    def C_m(self):
        return self.tsai_constant(self.ply.matrix)

    @property
    def E_2(self):
        #  Nijhof p91
        G_f, G_m, C_f, C_m, v_f, v_m, nu_f, nu_m, C = self.get('G_f, G_m, C_f, C_m, v_f, v_m, nu_f, nu_m, C')
        a1 = 2 * C_m * C_f + (v_m * C_m + v_f * C_f) * G_m
        b1 = 2 * (v_m * C_f + v_f * C_m) + G_m
        a2 = 2 * C_m * C_f + (v_m * C_m + v_f * C_f) * G_f
        b2 = 2 * (v_m * C_f + v_f * C_m) + G_f
        return (2 * (1 - v_m * nu_m - v_f * nu_f) *
                ((1 - C) * a1 / b1 + C * a2 / b2))

    @property
    def G_12(self):
        #  Nijhof p91
        G_f, G_m, C_f, C_m, v_f, v_m, C = self.get('G_f, G_m, C_f, C_m, v_f, v_m, C')
        a1 = v_m * G_m + v_f * G_f + G_f
        b1 = v_m * G_f + v_f * G_m + G_m
        a2 = v_m * G_m + v_f * G_f + G_m
        b2 = v_m * G_f + v_f * G_m + G_f
        return (1 - C) * G_m * a1 / b1 + C * G_f * a2 / b2

    @property
    def nu_12(self):
        # Nijhof p91
        G_f, G_m, C_f, C_m, v_f, v_m, nu_f, nu_m, C = self.get('G_f, G_m, C_f, C_m, v_f, v_m, nu_f, nu_m, C')
        a1 = 2 * C_m * C_f * (v_m * nu_m + v_f * nu_f) + (v_m * C_m * nu_m + v_f * C_f * nu_f) * G_m
        b1 = 2 * C_m * C_f + (v_m * C_m + (v_m * C_m + v_f * C_f) * G_m)
        a2 = 2 * C_m * C_f * (v_m * nu_m + v_f * nu_f) + (v_m * C_m * nu_m + v_f * C_f * nu_f) * G_f
        b2 = 2 * C_m * C_f + (v_m * C_m + (v_m * C_m + v_f * C_f) * G_f)
        return (1 - C) * a1 / b1 + C * a2 / b2


class HalpinTsai(RuleOfMixtures):
    pass


class TsaiHahn(RuleOfMixtures):
    pass


class ChristensenLo(RuleOfMixtures):
    pass


class Puck:
    pass


class Powell(RuleOfMixtures):
    @property
    def E_2(self):
        return 1 / ((self.v_f / self.E_f) + (self.v_m ** 1.25 / self.E_app) * (1 / (1 + 0.85 * self.v_f ** 2)))


class ThermalROM(Micromechanics):
    @property
    def alpha_11(self):
        E_f, E_m, v_f, v_m, alpha_f, alpha_m = self.get('E_f, E_m, v_f, v_m, alpha_f, alpha_m')
        return (alpha_f * E_f * v_f + alpha_m * E_m * v_m) / (E_f * v_f + E_m * v_m)

    @property
    def alpha_22(self):
        raise NotImplementedError()


class ThermalSchneider(ThermalROM):
    @property
    def alpha_22(self):
        E_f, E_m, v_f, v_m, alpha_f, alpha_m, nu_f, nu_m = self.get('E_f, E_m, v_f, v_m, alpha_f, alpha_m, nu_f, nu_m')
        a1 = 2 * (nu_m ** 2 - 1) * 1.1 * nu_f
        b1 = 1.1 * v_f * (2 * nu_m ** 2 + nu_m - 1) - (1 + nu_m)
        a2 = nu_m * E_f / E_m
        b2 = E_f / E_m + (1 - 1.1 * v_f) / (2.2 * v_f)
        return alpha_m - (alpha_m - alpha_f) * ((a1 / b1) - (a2 / b2))


class StrengthROM(Micromechanics):
    @property
    def sigmat11(self):
        E_f, E_m, v_f, v_m, sigma_ft, sigma_mt = self.get('E_f, E_m, v_f, v_m, sigma_ft, sigma_mt')

        fibrefailure = (v_f + v_m * E_m / E_f) * sigma_ft
        matrixfailure = (v_m + v_f * E_f / E_m) * sigma_mt

        return min(fibrefailure, matrixfailure)

    @sigmat11.setter
    def sigmat11(self, value):
        self.measured_properties['sigmat11'] = value

    @property
    def sigmac11(self):
        v_f = self.v_f
        v_m = self.matrixfraction
        E_m = self.matrix.modulus
        E_f = self.fibre.modulus
        sigma_ft = self.fibre.sigma_t
        sigma_mt = self.matrix.sigma_t
        sigma_mc = self.matrix.sigma_c

        def fibrefailure():
            return (v_f + v_m * E_m / E_f) * sigma_ft

        def matrixfailure():
            return (v_m + v_f * E_f / E_m) * sigma_mt

        return self._measured_or_calculated(strength, 'sigmac11')

    @sigmac11.setter
    def sigmac11(self, value):
        self.measured_properties['sigmac11'] = value

    @property
    def sigmat22(self):
        v_f = self.v_f
        E_m = self.matrix.modulus
        E_f = self.fibre.modulus
        sigma_mt = self.matrix.sigma_t

        def rosen():
            raise NotImplementedError()

        def mallick():
            # PK Mallick 'fibre reinforced composites materials' eq. 3.27, taken from Greszczuk
            a = 1 - v_f * (1 - E_m / E_f)
            b = 1 - math.sqrt(4 * v_f / math.pi) * (1 - E_m / E_f)
            return sigma_mt / (a / b)

        def matthews():
            # Frank L. Matthews 'Composite materials: engineering @ science'
            # Gatenkaas by Matthews (resin with holes)
            return sigma_mt * (1 - 2 * math.sqrt(v_f / math.pi))

    @property
    def sigmac22(self):
        # TODO: find better approximation (taken from Jan's sheet)
        return 1.2 * self.matrix.sigma_c

    @property
    def sigma6(self):
        # TODO: find better approximation (taken from Jan's sheet)
        return 0.5 * self.matrix.sigma_t

    def strength(self, direction, tensile_or_compressive=None):
        if direction == 6 and tensile_or_compressive is not None:
            raise AttributeError('shear strength is not tensile or compressive')
        if tensile_or_compressive is None:
            tensile_or_compressive = 'tensile'
        strength = {(1, 'tensile'): self.sigmat11,
                    (1, 'compressive'): self.sigmac11,
                    (2, 'tensile'): self.sigmat22,
                    (2, 'compressive'): self.sigmac22,
                    (6, 'tensile'): self.sigma6}
        return strength[(direction, tensile_or_compressive)]

    def unity_check(self, sigma1, sigma2, sigma6=0):
        s1t = self.strength(1, 'tensile')
        s1c = self.strength(1, 'compressive')
        s2t = self.strength(2, 'tensile')
        s2c = self.strength(2, 'compressive')
        if sigma1 > 0:
            s1 = s1t
        else:
            s1 = s1c
        if sigma2 > 0:
            s2 = s2t
        else:
            s2 = s2c
        s6 = self.sigma6


def quadratic(a, b, c=0):
    d = b ** 2 - 4 * a * c  # discriminant
    if d < 0:
        raise ValueError(f"Discriminant={d} < 0, no real solution to quadratic equation {a}x^2+{b}x+{c}=0")
    else:
        x1 = (-b + math.sqrt(d)) / (2 * a)
        x2 = (-b - math.sqrt(d)) / (2 * a)
    return x1, x2


class IsotropicMaterial:
    def __init__(self, name: str, modulus=0, shearmodulus=0, poissonratio=0, thermalcoefficient=0, rho=0,
                 sigma_t=0, sigma_c=0, S_xy=0, **kwargs):
        self.name = name
        self.modulus = modulus
        self.shearmodulus = shearmodulus
        self.poissonratio = poissonratio
        self.thermalcoefficient = thermalcoefficient
        self.rho = rho
        self.sigma_t = sigma_t
        self.sigma_c = sigma_c
        self.S_xy = S_xy


LayupLayer = namedtuple('LayupLayer', 'angle count material')


class Layup:
    """A Layup provides a textual representation of a laminate build up
    (nr. of layers in what material and in which direction). It does not
    contain the actual material or thicknesses"""

    def __init__(self, layup_string=None):
        self.parse(layup_string)

    def __iter__(self):
        return iter(self.layers)

    def __add__(self, other):
        if isinstance(other, Layup):
            layup = Layup()
            layup.layers = self.layers + other.layers
            return layup
        elif isinstance(other, str):
            layup = Layup()
            layup.layers = self.layers + self.parse(other)
            return layup
        else:
            raise TypeError(f"cannot add {self} and {other}")

    def ply_stack(self, fibres, matrix, ply_thickness, v_f, voidsfraction=0):
        stack = []
        for layer in self:
            for f in fibres:
                if f.name == layer.material:
                    fibre = f
                    break
            else:
                fibre = fibres[0]  # not found, assume first material
            stack.append(Ply(fibre, matrix, layer.count * ply_thickness, v_f, angle=layer.angle,
                             voidsfraction=voidsfraction))
        return stack

    @property
    def layer_count(self):
        return sum(l.count for l in self.layers)

    @property
    def angles(self):
        return set(l.angle for l in self.layers)

    @property
    def materials(self):
        return set(l.material for l in self.layers)

    @property
    def is_symmetrical(self):
        n = int(len(self.layers) / 2)
        for i in range(n):
            if self.layers[i] != self.layers[-(i + 1)]:
                return False
        return True

    @property
    def has_middle_layer(self):
        return self.is_symmetrical & len(self.layers) % 2 == 1

    @property
    def is_balanced(self):
        off_axis = {}
        for l in self.layers:
            if l.material not in off_axis:
                off_axis[l.material] = {}
            if l.angle not in (0, 90):
                if l.angle > 0:
                    count = l.count
                else:
                    count = -l.count
                angle = abs(l.angle)
                if angle not in off_axis[l.material]:
                    off_axis[l.material][angle] = count
                else:
                    off_axis[l.material][angle] += count
        for m in off_axis.values():
            for n in m.values():
                if n != 0:
                    return False
        return True

    def __str__(self):
        def iter_layers():
            if not self.is_symmetrical:
                for l in self.layers:
                    yield l
            else:
                n = len(self.layers)
                for l in self.layers[:int(n / 2) + 1]:
                    yield l

        s = '['

        for i, l in enumerate(iter_layers()):
            if i > 0:
                s += '/'
            s += f'{l.angle}'
            if l.count > 1:
                s += f'({l.count})'
            if l.material != self._material:
                s += '{' + l.material + '}'

        # print(l.material, l.count, l.angle)

        if self.has_middle_layer:  # middle layer
            s += 'M'

        s += ']'

        if self.is_symmetrical:
            s += 'S'

        if self._material is not None:
            s += '{' + self._material + '}'

        return s

    def parse(self, layup_string: str):
        # example: '[0/90(2){steel}M]S'
        # Symmetrical, middle layer
        # TODO: allow nesting

        if layup_string is None:
            return []
        if len(layup_string) == 0:
            return []

        layers = []
        # all-caps and no spaces
        layup_string.capitalize()
        layup_string = ''.join(layup_string.split())
        symmetric = False
        middle_layer = False
        material = None

        if layup_string.endswith('}'):  # default material specified
            layup_string = layup_string[:-1]
            material = layup_string.split('{')[-1]
            layup_string = '{'.join(layup_string.split('{')[:-1])

        if layup_string.endswith('S'):  # symmetric layup
            symmetric = True
            layup_string = layup_string[:-1]

        if layup_string.endswith(']'):
            layup_string = layup_string[:-1]

        if layup_string.startswith('['):
            layup_string = layup_string[1:]

        if layup_string.endswith('M'):  # middle layer
            if not symmetric:
                raise ValueError('Layup has middle layer but is not symmetric')
            middle_layer = True
            layup_string = layup_string[:-1]

        for substr in layup_string.split('/'):
            if substr.endswith('}'):  # material specified
                substr = substr[:-1]
                m = substr.split('{')[1]
                substr = substr.split('{')[0]
            else:
                m = material
            if substr.endswith(')'):
                substr = substr[:-1]
                angle = int(substr.split('(')[0])
                n = int(substr.split('(')[1])
            else:
                angle = int(substr)
                n = 1
            if abs(angle) > 90:
                raise ValueError('Angles must be between -90 and 90')
            layers.append(LayupLayer(angle=angle, count=n, material=m))

        if symmetric:
            if middle_layer:
                for l in reversed(layers[:-1]):
                    layers.append(l)
            else:
                for l in reversed(layers):
                    layers.append(l)

        self._material = material
        self.layers = layers


FabricLayer = namedtuple('FabricLayer', 'fibre angle mass')


class Fabric:
    """A Fabric provides a representation of a fabric build up
    (nr. of layers in what material and in which direction)"""

    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers = layers

    @property
    def rho(self):
        total_volume = sum([layer.mass / layer.fibre.rho for layer in self.layers])
        return self.mass / total_volume

    @property
    def mass(self):
        return sum([layer.mass for layer in self.layers])


class ModifiedRuleOfMixtures(RuleOfMixtures):
    @property
    def E_2(self):
        #  Kollar p442
        E_f, E_m = self.ply.symbols('E_f E_m')
        root = math.sqrt(self.v_f)
        E_b2 = root * E_f + (1 - root) * E_m
        return 1 / (root / E_b2 + (1 - root) / E_m)

    @property
    def G_12(self):
        #  Kollar p442
        G_f, G_m = self.ply.symbols('G_f G_m')
        root = math.sqrt(self.v_f)
        G_b12 = root * G_f + (1 - root) * G_m
        return 1 / (root / G_b12 + (1 - root) / G_m)

    @property
    def G_23(self):
        v_f, G_f, G_m = self.ply.symbols('v_f G_f G_m')
        #  Kollar p442
        root = math.sqrt(v_f)
        G_b23 = root * G_f + \
                (1 - root) * G_m
        return 1 / (root / G_b23 + (1 - root) / G_m)


class Ply:
    def __init__(self, fibre, matrix, t, v_f, *, angle=0, voidsfraction=0):
        # theory: any of 'rom', 'mod_rom', 'tsai'. 'halpin_tsai', 'tsai_hahn', 'christensen_lo', 'puck', 'powell'
        self.fibre = fibre
        self.matrix = matrix
        self.v_f = v_f
        self.t = t
        self.angle = angle
        self.voidsfraction = voidsfraction
        self.measured_properties = {'E_1': None, 'E_2': None, 'G_12': None, 'G_23': None, 'nu_12': None, 'nu_23': None}
        self.stiffness_model = RuleOfMixtures
        self.thermal_model = ThermalSchneider
        # self.strength_model = StiffnessPuck
        # self.failure_model = TsaiHill

    def copy(self, angle=None):
        if angle is None:
            angle = self.angle
        p = Ply(fibre=self.fibre, matrix=self.matrix, t=self.t, v_f=self.v_f, angle=angle, voidsfraction=self.voidsfraction)
        p.measured_properties = self.measured_properties
        p.stiffness_model = self.stiffness_model
        p.thermal_model = self.thermal_model
        # p.strength_model = self.strength_model
        # p.failure_model = self.failure_model
        return p

    @property
    def stiffness_model(self):
        return self._stiffness_model

    @stiffness_model.setter
    def stiffness_model(self, value):
        if not isinstance(value, Micromechanics):
            self._stiffness_model = value(self)
        else:
            self._stiffness_model = value

    def __str__(self):
        return f"{self.fibre}/{self.matrix} ply, t={self.t}, vf={self.v_f}, angle={self.angle}"

    @property
    def E_app(self):
        return self.matrix.modulus / (1 - self.matrix.poissonratio ** 2)  # apparent modulus

    @property
    def v_m(self):
        return 1.0 - self.v_f - self.voidsfraction  # Volume fraction of resin material

    @property
    def fibremass(self):
        return self.v_f * self.fibre.rho * self.t

    def _measured_or_calculated(self, attr):
        prop = self.measured_properties.get(attr, None)
        if prop is not None:
            return prop
        else:
            return getattr(self.stiffness_model, attr)

    @property
    def E_f(self):
        return self.fibre.modulus

    @property
    def E_m(self):
        return self.matrix.modulus

    @property
    def G_f(self):
        return self.fibre.shearmodulus

    @property
    def G_m(self):
        return self.matrix.shearmodulus

    @property
    def nu_f(self):
        return self.fibre.poissonratio

    @property
    def nu_m(self):
        return self.matrix.poissonratio

    @property
    def E_1(self):
        return self._measured_or_calculated('E_1')

    @E_1.setter
    def E_1(self, value):
        self.measured_properties['E_1'] = value

    @property
    def E_2(self):
        return self._measured_or_calculated('E_2')

    @E_2.setter
    def E_2(self, value):
        self.measured_properties['E_2'] = value

    @property
    def G_12(self):
        return self._measured_or_calculated('G_12')

    @G_12.setter
    def G_12(self, value):
        self.measured_properties['G_12'] = value

    @property
    def G_23(self):
        return self._measured_or_calculated('G_23')

    @G_23.setter
    def G_23(self, value):
        self.measured_properties['G_23'] = value

    @property
    def nu_12(self):
        return self._measured_or_calculated('nu_12')

    @nu_12.setter
    def nu_12(self, value):
        self.measured_properties['nu_12'] = value

    @property
    def nu_23(self):
        return self._measured_or_calculated('nu_23')

    @nu_23.setter
    def nu_23(self, value):
        self.measured_properties['nu_23'] = value

    def stiffness_matrix(self):
        E_1 = self.E_1
        E_2 = self.E_2
        G_12 = self.G_12
        nu_12 = self.nu_12

        Q11 = E_1 ** 2 / (E_1 - nu_12 * E_2)
        Q12 = nu_12 * E_1 * E_2 / (E_1 - nu_12 ** 2 * E_2)
        Q22 = E_1 * E_2 / (E_1 - nu_12 ** 2 * E_2)
        Q66 = G_12

        return np.array([[Q11, Q12, 0],
                         [Q12, Q22, 0],
                         [0, 0, Q66]])

    def global_stiffness_matrix(self, angle=0):
        # E_m = self.matrix.modulus
        # E_f = self.fibre.modulus

        # alpha1_f = self.fibre.thermalcoefficient
        # alpha_m = self.matrix.thermalcoefficient

        # rho = self.rho
        # vf = self.v_f
        # vm = self.matrixfraction

        # alpha1 = (alpha1_f * E_f * vf + alpha_m * E_m * vm) / E_1
        # alpha2 = alpha_m  # This is not 100% accurate, but simple.
        # alphax = alpha1 * m2 + alpha2 * n2
        # alphay = alpha1 * n2 + alpha2 * m2
        # alphaxy = 2 * (alpha1 - alpha2) * m * n

        # http://wstein.org/edu/2010/480b/projects/05-lamination_theory/A%20summary%20of%20Classical%20Lamination%20Theory.pdf
        # p 70
        # The powers of the sine and cosine are often used later.
        a = math.radians(self.angle + angle)
        c, s = math.cos(a), math.sin(a)

        Q = self.stiffness_matrix()
        Q11, Q12, Q21, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 0], Q[1, 1], Q[2, 2]

        Q_11 = Q11 * c ** 4 + 2 * (Q12 + 2 * Q66) * c ** 2 * s ** 2 + Q22 * s ** 4
        Q_12 = Q12 * (c ** 4 + s ** 4) + (Q11 + Q22 - 4 * Q66) * c ** 2 * s ** 2
        Q_16 = (Q11 - Q12 - 2 * Q66) * c ** 3 * s - (Q22 - Q12 - 2 * Q66) * c * s ** 3
        Q_22 = Q11 * s ** 4 + 2 * (Q12 + 2 * Q66) * c ** 2 * s ** 2 + Q22 * c ** 4
        Q_26 = (Q11 - Q12 - 2 * Q66) * c * s ** 3 - (Q22 - Q12 - 2 * Q66) * c ** 3 * s
        Q_66 = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * c ** 2 * s ** 2 + Q66 * (c ** 4 + s ** 4)

        return np.array([[Q_11, Q_12, Q_16],
                         [Q_12, Q_22, Q_26],
                         [Q_16, Q_26, Q_66]])

    def laminate_to_ply_transformation_matrix(self, angle=0):
        # p178, (8.20)
        a = math.radians(self.angle + angle)
        c, s = math.cos(a), math.sin(a)
        return np.array([[c ** 2, s ** 2, c * s],
                         [s ** 2, c ** 2, - c * s],
                         [-2 * s * c, 2 * s * c, c ** 2 - s ** 2]])


LaminateProperties = namedtuple('LaminateProperties', 'E_x E_y G_xy nuxy nuyx')


class Laminate:
    def __init__(self, stack_or_layup, *, fibres=None, matrix=None, ply_thickness=1e-3, v_f=0.5,
                 voidsfraction=0):
        if isinstance(stack_or_layup, str):  # layup string
            layup = Layup(stack_or_layup)
            self.stack = layup.ply_stack(fibres, matrix, ply_thickness, v_f, voidsfraction)
        elif isinstance(stack_or_layup, Layup):  # Layup object
            self.stack = stack_or_layup.ply_stack(fibres, matrix, ply_thickness, v_f, voidsfraction)
        # elif:  # fabric stack
        #     def fabrics_to_laminate(fabrics, matrix, v_f, voidsfraction=0):
        #         stack = []
        #         for fabric in fabrics:
        #             for layer in fabric.layers:
        #                 thickness = layer.areamass / (v_f * layer.fibre.rho)
        #                 ply = Ply(layer.fibre, matrix, thickness, angle=layer.angle,
        #                           v_f=v_f, voidsfraction=voidsfraction)
        #                 stack.append(ply)
        #         return Laminate(stack)
        else:  # stack
            self.stack = stack_or_layup if stack_or_layup is not None else []

    def __iter__(self):
        return iter(self.stack)

    def __add__(self, other):
        # TODO: figure out angle of stcaked laminates
        stack = self.stack + other.stack
        return Laminate(stack)

    def distance(self, E_x_min=0, E_y_min=0, G_xy_min=0):
        prop = self.analyze()
        if prop.E_x < E_x_min or prop.E_y < E_y_min or prop.G_xy < G_xy_min:
            # does not meet requirements
            sign = 1
        else:
            sign = -1
        d = ((E_x_min - prop.E_x) / E_x_min) ** 2 + \
            ((E_y_min - prop.E_y) / E_y_min) ** 2 + \
            ((G_xy_min - prop.G_xy) / G_xy_min) ** 2
        return sign * math.sqrt(d)

    @property
    def matrix(self):
        return self.stack[0].matrix

    @property
    def t(self):
        return sum([ply.t for ply in self.stack])

    @property
    def rho(self):
        return sum([ply.rho * ply.t for ply in self.stack]) / self.t

    @property
    def v_f(self):
        return sum([ply.v_f * ply.t for ply in self.stack]) / self.t

    def iter_ply_offset(self):
        zs = - self.t / 2
        for ply in self.stack:
            z = zs + ply.t
            z2 = (z ** 2 - zs ** 2) / 2
            z3 = (z ** 3 - zs ** 3) / 3
            yield ply, z, z2, z3
            zs = z

    def ABD_matrix(self, angle=0):
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        for i, j in product(range(3), range(3)):
            for ply, z, z2, z3 in self.iter_ply_offset():
                Q_ = ply.global_stiffness_matrix(angle=angle)
                t = ply.t
                A[i, j] += Q_[i, j] * t
                B[i, j] += Q_[i, j] * z2
                D[i, j] += Q_[i, j] * z3
        ABD = np.bmat([[A, B], [B, D]])

        # # Thermal
        # # Ntx, Nty, Ntxy = 0.0, 0.0, 0.0

        # Finish the matrices, discarding very small numbers in ABD.
        for i in range(6):
            for j in range(6):
                if math.fabs(ABD[i, j]) < 1e-9:
                    ABD[i, j] = 0

        return ABD

    def abd_matrix(self, angle=0):
        ABD = self.ABD_matrix(angle)
        return np.linalg.inv(ABD)

    def polar_plot(self, width=10, height=8, step_size=5, E_x=True, E_y=True, figure=None, title=None, range_label=""):
        if figure is None:
            figure = plt.figure(figsize=(width, height))
            ax = figure.add_subplot(111, projection='polar')
        else:
            ax = figure.gca()

        data = {}
        props = {angle: self.analyze(angle=angle) for angle in range(0, 360 + step_size, step_size)}
        theta = 2 * np.pi / 360 * np.array(list(a for a, p in props.items()))
        data['E_x'] = np.array(list(p.E_x for a, p in props.items())) / 1e9
        data['E_y'] = np.array(list(p.E_y for a, p in props.items())) / 1e9

        if E_x:
            ax.plot(theta, data['E_x'], label="$E_x$ [GPa]" + range_label )
        if E_y:
            ax.plot(theta, data['E_y'], label="$E_y$ [GPa]" + range_label)


        # ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
        ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
        ax.grid(True)
        ax.set_theta_zero_location("N")

        ax.legend()
        if title is None:
            ax.set_title("Polar plot", va='bottom')
        else:
            ax.set_title(title, va='bottom')
        return figure

    def analyze(self, load=None, angle=0):
        ABD = self.ABD_matrix(angle)
        dABD = np.linalg.det(ABD)
        dt1 = np.linalg.det(ABD[1:6, 1:6])
        dt2 = np.linalg.det(np.delete(np.delete(ABD, 1, 0), 1, 1))
        dt3 = np.linalg.det(np.delete(np.delete(ABD, 2, 0), 2, 1))
        dt4 = np.linalg.det(np.delete(np.delete(ABD, 0, 0), 1, 1))
        dt5 = np.linalg.det(np.delete(np.delete(ABD, 1, 0), 0, 1))

        t = self.t

        E_x = (dABD / (dt1 * t))
        E_y = (dABD / (dt2 * t))
        G_xy = (dABD / (dt3 * t))

        nuxy = dt4 / dt1
        nuyx = dt5 / dt2

        # if load is not None:
        #     self.analyze_load(load)

        return LaminateProperties(E_x=E_x, E_y=E_y, G_xy=G_xy, nuxy=nuxy, nuyx=nuyx)

        # Calculate unit thermal stress resultants.
        # Hyer:1998, p. 445
        # Ntx += ( l.Q11 * l.alphax + l.Q12 * l.alphay + l.Q16 * l.alphaxy) * l.thickness
        # Nty += (l.Q12 * l.alphax + l.Q22 * l.alphay + l.Q26 * l.alphaxy) * l.thickness
        # Ntxy += (l.Q16 * l.alphax + l.Q26 * l.alphay + l.Q66 * l.alphaxy) * l.thickness

        # Calculate the engineering properties.
        # Nettles:1994, p. 34 e.v.

        # abd = self.abd_matrix()
        # non-symmetric laminates
        # Calculate the coefficients of thermal expansion.
        # Technically only valid for a symmetric laminate!
        # Hyer:1998, p. 451, (11.86)
        # alphax = abd[0, 0] * Ntx + abd[0, 1] * Nty + abd[0, 2] * Ntxy
        # alphay = abd[1, 0] * Ntx + abd[1, 1] * Nty + abd[1, 2] * Ntxy

    def analyze_load(self, load, angle=0):
        abd = self.abd_matrix(angle)
        try:
            loadvector = load.vector
        except AttributeError:
            loadvector = load
        strain_vector = abd.dot(loadvector).transpose()
        epsilon0 = strain_vector[0:3]  # in-plane strains
        kappa = strain_vector[3:6]  # curvatures
        ply_strains = []
        ply_stresses = []

        for ply, z, z2, z3 in self.iter_ply_offset():
            T = ply.laminate_to_ply_transformation_matrix()
            Q = ply.stiffness_matrix()
            t = ply.t
            strain = T.dot(epsilon0 + z * kappa)
            stress = Q.dot(strain)
            # [sigma_xx, sigma_yy, tau_xy] -> failure?
            print('RF', ply.unity_check(sigma1=stress[0], sigma2=stress[1], sigma6=stress[2]))
            ply_strains.append(strain)
            ply_stresses.append(stress)
        return ply_stresses, ply_strains


def build_laminate(ply, angles):
    stack = [ply.copy(angle=a) for a in angles]
    return Laminate(stack)


def insert_mid(l, v):
    n = int(len(l) / 2)
    if len(l) % 2 == 0:
        return l[:n] + v + l[n:]
    else:
        return l[:n + 1] + v + l[n:]


def potential_laminate(ply, angles, next_angle, symmetrical=True, balanced=True):
    if balanced and 0 < abs(next_angle) < 90:
        if symmetrical:
            insert_angles = [next_angle, -next_angle, -next_angle, next_angle]
        else:
            insert_angles = [next_angle, -next_angle]
    else:
        insert_angles = [next_angle]

    if symmetrical and len(angles) == 1:
        angles = angles + angles
    elif symmetrical:
        angles = insert_mid(angles, insert_angles)
    else:
        angles = angles + insert_angles
    return build_laminate(ply, angles), angles


def optimize_laminate(ply, possible_angles, E_x_min=0, E_y_min=0, G_xy_min=0, n_max=20, symmetrical=True, balanced=True, plot=False):
    if plot:
        figure = plt.figure()
        ax = figure.add_subplot(111, projection='polar')
    angles = []
    done = False
    n = 0
    while n < n_max and not done:
        n += 1
        best_distance = None
        best_laminate = None
        for next_angle in possible_angles:
            next_laminate, next_angles = potential_laminate(ply, angles, next_angle, symmetrical=symmetrical, balanced=balanced)
            d = next_laminate.distance(E_x_min=E_x_min, E_y_min=E_y_min, G_xy_min=G_xy_min)
            if best_distance is None:
                best_distance = d
                best_laminate = next_laminate
                best_angles = next_angles
            else:
                if d < best_distance:
                    best_distance = d
                    best_laminate = next_laminate
                    best_angles = next_angles
        angles = best_angles
        if plot:
            best_laminate.polar_plot(figure=figure, range_label="({})".format(n), E_y=False)
        if best_distance < 0:
            print("found!", best_angles)
            if plot:
                ax = figure.gca()
                ax.plot([0], [E_x_min / 1e9], marker="x", color="k")
                ax.plot([90 / 180 * np.pi], [E_y_min / 1e9], marker="x", color="k")
                ax.plot([180 / 180 * np.pi], [E_x_min / 1e9], marker="x", color="k")
                ax.plot([270 / 180 * np.pi], [E_y_min / 1e9], marker="x", color="k")
                return best_laminate, figure
            else:
                return best_laminate
    raise AttributeError("no laminate found")

# class Load:
#     def __init__(self, Nxx=0, Nyy=0, Nxy=0, Mxx=0, Myy=0, Mxy=0):
#         self.Nxx = Nxx
#         self.Nyy = Nyy
#         self.Nxy = Nxy
#         self.Mxx = Mxx
#         self.Myy = Myy
#         self.Mxy = Mxy
#
#     @property
#     def vector(self):
#         return np.array([self.Nxx, self.Nyy, self.Nxy, self.Mxx, self.Myy, self.Mxy])
#
#     @vector.setter
#     def vector(self, vector):
#         self.Nxx = vector[0]
#         self.Nyy = vector[1]
#         self.Nxy = vector[2]
#         self.Mxx = vector[3]
#         self.Myy = vector[4]
#         self.Mxy = vector[5]


# class FlatPlate:
#     def __init__(self, thickness: float, material: Laminate, dimensions: tuple, load: Load):
#         self.thickness = thickness
#         self.material = material
#         self.dimensions = dimensions
#         self.load = load
#
#     def analyze(self):
#         pass

# fabrics = {'090': fabric(fibres['e-glass'], 1.2, (0, 90)),
#            '+-45': fabric(fibres['e-glass'], 1.2, (45, -45)),
#            'ud': fabric(fibres['e-glass'], 1.2, (0, ))}

# laminates = {'75/25-0/90': fabrics_to_laminate([fabrics['ud'], fabrics['090']], resins['synolite'], v_f=0.54)}

# l = laminates['75/25-0/90']
# l.update()

class MaterialDB:
    def __init__(self, filename="materials.db"):
        self.filename = filename

    def get_isotropic(self, name):
        db = sqlite3.connect(self.filename)
        # id name modulus shearmodulus poissonratio rho
        # sigma_t sigma_c
        # S_xy thermalcoefficient
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        cursor.execute("SELECT * FROM isotropic WHERE name=?", (name,))
        ## fetchone() returns a dict with keys as column names
        return IsotropicMaterial(**cursor.fetchone())

    def all_isotropic_materials(self):
        db = sqlite3.connect(self.filename)
        # id name modulus shearmodulus poissonratio rho
        # sigma_t sigma_c
        # S_xy thermalcoefficient
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        cursor.execute("SELECT * FROM isotropic")
        ## fetchone() returns a dict with keys as column names
        materials = []
        for data in cursor.fetchall():
            materials.append(IsotropicMaterial(**data))
        return materials

    def create(self):
        db = sqlite3.connect(self.filename)
        cursor = db.cursor()
        qry = open('create_table_isotropic.sql', 'r').read()
        cursor.execute(qry)
        db.commit()
        db.close()


def main():
    db = MaterialDB()
    layup_string = '[-45/45{steel}/90(2){steel}M]S{eglass}'
    layup_string = '[0(3)/90/-45/45]S{eglass}'
    # print(layup_string)
    layup = Layup(layup_string)
    # print(layup)
    # for l in layup:
    #     print(l)

    # print(layup.layer_count)
    # print(layup.angles)
    # print(layup.is_symmetrical)
    # print(layup.is_balanced)
    # print(layup.materials)

    M = db.get_isotropic
    lam = Laminate(layup_string, fibres=[M('eglass'), M('steel')],
                   matrix=M('epoxy'), ply_thickness=0.2, v_f=0.55)

    # prop = lam.analyze()

    all = db.all_isotropic_materials()

    for l in lam.stack:
        print(l)

    ply = lam.stack[0]
    E_x_min = 28e9
    E_y_min = 13e9
    G_xy_min = 4.5e9
    opt_lam, fig = optimize_laminate(ply, [0, 90, 45, -45], E_x_min=E_x_min, E_y_min=E_y_min, G_xy_min=G_xy_min, plot=True)
    opt_lam_prop = opt_lam.analyze()

    # fig = opt_lam.polar_plot(step_size=2)
    plt.show()

    print(opt_lam.distance(E_x_min=E_x_min, E_y_min=E_y_min, G_xy_min=G_xy_min))

    # l1 = Laminate.from_db('e-glass', 'polyester', 0.55, [0, 90, 45, -45], symmmetric=True, middle_layer=False, ply_thickness=0.2)

    # create_db()


if __name__ == '__main__':
    main()
