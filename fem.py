import math

import numpy as np
import scipy.linalg


class Material:
    def __init__(self, E, rho, G=1e12):
        self.E = E
        self.rho = rho
        self.G = G


class Section:
    def __init__(self, material, A, I, As=False):
        self.material = material
        self.A = A
        self.I = I
        self.As = A
        if As:
            self.As = As

    @property
    def mass(self):
        return self.material.rho * self.A

    @property
    def EI(self):
        return self.material.E * self.I

    @property
    def GA(self):
        return self.material.G * self.As

    @property
    def EA(self):
        return self.material.E * self.A


class DistributedLoad:
    def __init__(self, q=(0, 0), a=(0, 1)):
        self.q = q
        self.a = a

    def slope(self):
        return self.q[1] - self.q[0] / (self.a[1] - self.a[0])

    def value_at(self, a):
        if not self.a[0] < a < self.a[1]:
            return 0
        return self.slope * (a - self.a[0]) + self.a[0]

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if isinstance(value, float) or isinstance(value, int):
            value = (value, value)
        self._q = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value[0] > value[1]:
            value = (value[1], value[0])
        self._a = value


class PointLoad:
    def __init__(self, Fx=0, Fy=0, M=0, a=0):
        if isinstance(Fx, tuple):
            Fx, Fy, M = Fx

        if not 0 <= a <= 1:
            raise ValueError('position a must be between 0 and 1')

        self.Fx = Fx
        self.Fy = Fy
        self.M = M
        self.a = a

    def value_at(self, a):
        if math.isclose(self.a, a):
            return (self.Fx, self.Fy, self.M)
        else:
            return (0, 0, 0)

    @property
    def vector(self):
        return np.array([self.Fx, self.Fy, self.M])

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __add__(self, other):
        if not isinstance(other, PointLoad):
            other = PointLoad(other)
        self.Fx += other.Fx
        self.Fy += other.Fy
        self.M += other.M
        return self

    def __sub__(self, other):
        if not isinstance(other, PointLoad):
            other = PointLoad(other)
        self.Fx -= other.Fx
        self.Fy -= other.Fy
        self.M -= other.M
        return self


class Node:
    FIXED = (True, True, True)
    FIXEDX = (True, False, False)
    FIXEDY = (False, True, False)
    HINGED = (True, True, False)
    FREE = (False, False, False)

    def __init__(self, pos):
        self.pos = pos
        self.p_loads = []
        self.constraints = Node.FREE

    def add_load(self, P):
        if not isinstance(P, PointLoad):
            P = PointLoad(P)
        self.p_loads.append(P)

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def load_vector(self):
        P = PointLoad()
        for p in self.p_loads:
            P += p
        return P.vector

    def fix(self):
        self.constraints = Node.FIXED

    def fix_x(self):
        self.constraints = Node.FIXEDX

    def fix_y(self):
        self.constraints = Node.FIXEDY

    def hinge(self):
        self.constraints = Node.HINGED


class Element:
    def __init__(self, n1, n2, section):
        self.n1 = n1
        self.n2 = n2
        self.section = section

        self.q_loads = []
        self.p_loads = []

    def position_at(self, a):
        return (self.n2.x + a * (self.n2.x - self.n1.x),
                self.n2.y + a * (self.n2.y - self.n1.y))

    def p_load_at(self, a):
        p_loads = [p.value_at(a) for p in self.p_loads]
        Fx = sum(l[0] for l in p_loads)
        Fy = sum(l[1] for l in p_loads)
        M = sum(l[2] for l in p_loads)
        return Fx, Fy, M

    def q_load_at(self, a):
        return sum([q.value_at(a) for q in self.q_loads])

    def mesh(self, system=None):
        pos = [0, 1]
        pos += [p.a for p in self.p_loads]
        pos += [q.a[0] for q in self.q_loads]
        pos += [q.a[1] for q in self.q_loads]
        pos = list(sorted(set(pos)))

        if system is None:
            system = System()

        meshed = []
        for i in range(len(pos) - 1):
            a1, a2 = pos[i], pos[i + 1]
            # find loads acting at this position
            Fx, Fy, M = self.p_load_at(a1)
            q1 = self.q_load_at(a1)
            q2 = self.q_load_at(a2)

            element = system.add_element(
                self.position_at(a1), self.position_at(a2), self.section)
            element.n1.add_load(P=(Fx, Fy, M))
            # TODO: q1 / q2
            element.add_load(q=q1)
            meshed.append(element)

        # move loads at end node to mesh node
        Fx, Fy, M = self.p_load_at(a=1)
        # only add load to node if it wasnt already loaded from another 'element to be meshed'
        if len(meshed[-1].n2.p_loads) == 0:
            meshed[-1].n2.add_load(P=(Fx, Fy, M))

        # copy constraints
        meshed[0].n1.constraints = self.n1.constraints
        meshed[-1].n2.constraints = self.n2.constraints

        return system

    def add_load(self, q=None, F=None, a=0):
        if isinstance(q, DistributedLoad):
            self.q_loads.append(q)
            return q
        if isinstance(q, PointLoad):
            self.p_loads.append(q)
            return q
        if F is not None:
            Fx, Fy, M = F
            self.p_loads.append(PointLoad(Fx=Fx, Fy=Fy, M=M, a=a))

    @property
    def a_abs(self):
        return self.a * self.length

    @property
    def mass(self):
        return self.section.mass * self.length

    @property
    def length(self):
        return math.sqrt((self.n2.x - self.n1.x) ** 2 + (self.n2.y - self.n1.y) ** 2)

    @property
    def alpha(self):
        return math.atan2(self.n2.y - self.n1.y, self.n2.x - self.n1.x)

    @property
    def shear_factor(self):
        return 12 * self.section.EI / (self.section.GA * self.length ** 2)

    @property
    def transformation_matrix(self):
        r"""Determines the element's transformation matrix from the local
        coordinate system to the global coordinate system defined in cs

        .. math::
           T =
           \begin{bmatrix}
           C& S& 0&  0& 0& 0 \\
           -S& C& 0&  0& 0& 0 \\
           0&  0& 1&  0& 0& 0 \\
           0&  0& 0&  C& S& 0 \\
           0&  0& 0& -S& C& 0 \\
           0&  0& 0&  0& 0& 1 \\
           \end{bmatrix}

        with:

        .. math::
           C=cos(\alpha)=\frac{dx}{L}
           S=sin(\alpha)=\frac{dy}{L}

        Returns: A numpy array representing the transformation matrix
        (a 6x6 matrix)

        """
        c = (self.n2.x - self.n1.x) / self.length  # cos
        s = (self.n2.y - self.n1.y) / self.length  # sin
        return np.array([
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1]])

    @property
    def mass_matrix(self):
        r"""Determines mass matrix for the element in it's local coordinate
        system. Includes shear deformation based on the shear area of the
        elements cross section.

        phi is the ratio of bending and shear rigidities

        If the shear area is zero or None, or shear==False, shear effects are
        neglected (phi=0)

        Mass matrix is described in document "Structural Elements.pdf"

        Helper values:

        .. math::
           \phi=\frac{12EI}{A_sGL^2}

        Now we can determine the local mass matrix:

        .. math::
           M_{local}=\frac{\rho\A\L}{840}
           \begin{bmatrix}

           \end{bmatrix}

        Returns: A NumPy array representing the local mass matrix
        (a 6x6 matrix)

        :param shear: boolean, if True, account for shear effects

        """
        m = self.section.mass
        l = self.length
        phi = self.shear_factor

        a1 = 280
        d1 = a4 = 140
        b2 = 312 + 588 * phi + 280 * phi ** 2
        c2 = b3 = (44 + 77 * phi + 35 * phi ** 2) * l
        e2 = b5 = 108 + 252 * phi + 175 * phi ** 2
        f2 = b6 = -(26 + 63 * phi + 35 * phi ** 2) * l
        c3 = (8 + 14 * phi + 7 * phi ** 2) * l ** 2
        e3 = c5 = (26 + 63 * phi + 35 * phi ** 2) * l
        f3 = c6 = -(6 + 14 * phi + 7 * phi ** 2) * l ** 2
        d4 = 280
        e5 = 312 + 588 * phi + 280 * phi ** 2
        f5 = e6 = -(44 + 77 * phi + 35 * phi ** 2) * l
        f6 = (8 + 14 * phi + 7 * phi ** 2) * l ** 2
        return np.array([
            [a1, 0, 0, d1, 0, 0],
            [0, b2, c2, 0, e2, f2],
            [0, b3, c3, 0, e3, f3],
            [a4, 0, 0, d4, 0, 0],
            [0, b5, c5, 0, e5, f5],
            [0, b6, c6, 0, e6, f6]]) * (m * l / 840)

    @property
    def stiffness_matrix(self):
        r"""Determines stiffness matrix for the element in it's local
        coordinate system. Includes shear deformation based on the shear
        area of the elements cross section.

        phi is the ratio of bending and shear rigidities

        If the shear area is zero or None, or shear==False, shear effects are
        neglected (phi=0)

        Helper values:

        .. math::
           \phi=\frac{12EI}{A_sGL^2}
           C_1=\frac{AE}{L}
           C_2=\frac{EI}{L^3(1+\phi)}

        Now we can determine the local stiffness matrix:

        .. math::
           K_{local} =
           \begin{bmatrix}
           C_1&     0&            0&      -C_1&      0&           0     \\
           0&     12C_2&       6C_2L&       0&   -12C_2&      6C_2L     \\
           0&     6C_2L&  (4+\phi)C_2L^2&   0&   -6C_2L& (2-\phi)C_2L^2 \\
           -C_1&    0&            0&       C_1&      0&           0     \\
           0&     -12C_2&      -6C_2L&      0&    12C_2&      -6C_2L    \\
           0&     6C_2L&  (2-\phi)C_2L^2&   0&   -6C_2L& (4+\phi)C_2L^2 \\
           \end{bmatrix}

        Returns: A NumPy array representing the local stiffness matrix
        (a 6x6 matrix)

        :param shear: if True, account for shear deformations

        """
        l = self.length
        phi = self.shear_factor

        c_1 = self.section.EA / l
        c_2 = self.section.EI / (l ** 3 * (1 + phi))
        return np.array([
            [c_1, 0, 0, -c_1, 0, 0],
            [0, 12 * c_2, 6 * c_2 * l, 0, -12 * c_2, 6 * c_2 * l],
            [0, 6 * c_2 * l, (4 + phi) * c_2 * l ** 2, 0, -6 * c_2 * l,
             (2 - phi) * c_2 * l ** 2],
            [-c_1, 0, 0, c_1, 0, 0],
            [0, -12 * c_2, -6 * c_2 * l, 0, 12 * c_2, -6 * c_2 * l],
            [0, 6 * c_2 * l, (2 - phi) * c_2 * l ** 2, 0, -6 * c_2 * l,
             (4 + phi) * c_2 * l ** 2]])

    @property
    def load_vector(self):
        M1 = M2 = V1 = V2 = 0
        for ql in self.q_loads:
            if not math.isclose(q.a[0], 0) or not math.isclose(q.a[1], 0):
                raise ArithmeticError('can determine load factor for an incomplete meshed element')
            q1, q2 = ql.q
            L = self.length
            M1 += 1 / 12 * q1 * L ** 2 + 1 / 30 * (q2 - q1) * L ** 2
            M2 += 1 / 12 * q1 * L ** 2 + 1 / 20 * (q2 - q1) * L ** 2
            V1 += -1 / 2 * q1 * L - 3 / 20 * (q2 - q1) * L
            V2 += -1 / 2 * q1 * L - 7 / 20 * (q2 - q1) * L
        return np.array([0, V1, -M1, 0, V2, M2])

    def transformed_matrix(self, matrix):
        tau = self.transformation_matrix
        return tau.T.dot(matrix).dot(tau)

    def transformed_vector(self, vector):
        tau = self.transformation_matrix
        return tau.T.dot(vector)

    @property
    def transformed_mass_matrix(self):
        return self.transformed_matrix(self.mass_matrix)

    @property
    def transformed_stiffness_matrix(self):
        return self.transformed_matrix(self.stiffness_matrix)

    @property
    def transformed_load_vector(self):
        return self.transformed_vector(self.load_vector)


class System:
    def __init__(self):
        self.elements = []
        self.fixed = []
        self.hinged = []
        self.nodes = []

    @property
    def bounding_box(self):
        if len(self.nodes) == 0:
            return ((0, 0), (0, 0))
        x_max, y_max = x_min, y_min = self.nodes[0].x, self.nodes[0].y
        for n in self.nodes:
            if n.x < x_min:
                x_min = n.x
            if n.x > x_max:
                x_max = n.x
            if n.y < y_min:
                y_min = n.y
            if n.y > y_max:
                y_max = n.y
        return ((x_min, y_min), (x_max, y_max))

    @property
    def fixed_nodes(self):
        return [n for n in self.nodes if n.constraints == Node.FIXED]

    @property
    def hinged_nodes(self):
        return [n for n in self.nodes if n.constraints == Node.HINGED]

    @property
    def free_nodes(self):
        return [n for n in self.nodes if n.constraints == Node.FREE]

    @property
    def fixed_x_nodes(self):
        return [n for n in self.nodes if n.constraints == Node.FIXEDX]

    @property
    def fixed_y_nodes(self):
        return [n for n in self.nodes if n.constraints == Node.FIXEDY]

    def get_node(self, p):
        if p in self.nodes:
            return p
        for n in self.nodes:
            if n.pos == p:
                return n
        raise ValueError()

    def get_element(self, p1, p2):
        n1 = self.get_node(p1)
        n2 = self.get_node(p2)

        for e in self.elements:
            if e.n1 in (n1, n2) and e.n2 in (n1, n2):
                return e
        raise ValueError()

    def add_node(self, p):
        # return node if already exists
        try:
            return self.get_node(p)
        except ValueError:
            pass
        # otherwise create and add new node
        if isinstance(p, Node):
            n = p
        else:
            n = Node(p)
        self.nodes.append(n)
        return n

    def add_element(self, p1, p2, section):
        n1 = self.add_node(p1)
        n2 = self.add_node(p2)

        b = Element(n1, n2, section)
        self.elements.append(b)
        return b

    def fix(self, p):
        self.get_node(p).fix()

    def fix_x(self, p):
        self.get_node(p).fix_x()

    def fix_y(self, p):
        self.get_node(p).fix_y()

    def hinge(self, p):
        self.get_node(p).hinge()

    def free(self, p):
        self.get_node(p).free()

    @property
    def ndofs(self):
        return 3 * len(self.nodes)

    @property
    def stiffness_matrix(self):
        K = np.zeros((self.ndofs, self.ndofs))

        for e in self.elements:
            k = e.transformed_stiffness_matrix
            self.assemble_matrix(K, k, e)
        return K

    @property
    def mass_matrix(self):
        M = np.zeros((self.ndofs, self.ndofs))
        for e in self.elements:
            m = e.transformed_mass_matrix
            self.assemble_matrix(M, m, e)
        return M

    def assemble_matrix(self, global_matrix, local_matrix, element):
        i1 = 3 * self.nodes.index(element.n1)
        i2 = 3 * self.nodes.index(element.n2)
        global_matrix[i1:i1 + 3, i1:i1 + 3] += local_matrix[0:3, 0:3]
        global_matrix[i1:i1 + 3, i2:i2 + 3] += local_matrix[0:3, 3:6]
        global_matrix[i2:i2 + 3, i1:i1 + 3] += local_matrix[3:6, 0:3]
        global_matrix[i2:i2 + 3, i2:i2 + 3] += local_matrix[3:6, 3:6]

    def assemble_vector(self, global_vector, local_vector, obj):
        try:
            # 2D element
            i1 = 3 * self.nodes.index(obj.n1)
            i2 = 3 * self.nodes.index(obj.n2)
            global_vector[i1:i1 + 3] += local_vector[0:3]
            global_vector[i2:i2 + 3] += local_vector[3:6]
        except AttributeError:
            # node
            i = 3 * self.nodes.index(obj)
            global_vector[i:i + 3] += local_vector[0:3]

    @property
    def reduced_stiffness_matrix(self):
        return self.reduced(self.stiffness_matrix)

    @property
    def reduced_mass_matrix(self):
        return self.reduced(self.mass_matrix)

    @property
    def load_vector(self):
        F = np.zeros(self.ndofs)
        for e in self.elements:
            f = e.transformed_load_vector
            self.assemble_vector(F, f, e)
        for n in self.nodes:
            f = n.load_vector
            self.assemble_vector(F, f, n)
        return F

    @property
    def reduced_load_vector(self):
        return self.reduced(self.load_vector)

    @property
    def constrained_dofs(self):
        constrained = []
        i = 0
        for n in self.nodes:
            for c in n.constraints:
                if c:  # this dof is constrained -> add to constraints list
                    constrained.append(i)
                i += 1
        return np.array(sorted(constrained))

    @property
    def free_dofs(self):
        c = self.constrained_dofs
        return np.array([dof for dof in range(self.ndofs) if not dof in c])

    def reduced(self, matrix):
        if len(matrix.shape) > 1:  # matrix
            for i in [0, 1]:
                matrix = np.delete(matrix, self.constrained_dofs, axis=i)
        else:  # vector
            matrix = np.delete(matrix, self.constrained_dofs)
        return matrix

    def expanded(self, matrix):
        if len(matrix.shape) > 1:
            raise NotImplementedError()
        else:
            for dof in self.constrained_dofs:
                matrix = np.insert(matrix, dof, 0)
        return matrix

    def mesh(self):
        meshed_system = System()
        for element in self.elements:
            element.mesh(meshed_system)
        return meshed_system

    def solve(self):
        meshed = self.mesh()
        K_red = self.reduced_stiffness_matrix
        M_red = self.reduced_mass_matrix
        F_red = self.reduced_load_vector

        K = self.stiffness_matrix
        F = self.load_vector

        # find the natural frequencies
        evals, evecs = scipy.linalg.eigh(K_red, M_red)
        frequencies = np.sqrt(evals)

        # calculate the static displacement of each element for the reduced system
        X_red = np.linalg.inv(K_red).dot(F_red)

        # expand displacement vector to non-reduced system
        X = self.expanded(X_red)

        # find all nodal forces
        F_nodal = K.dot(X)

        # find reaction forces
        R = F_nodal - F

        # force constrained dofs to be zero (no reaction force there)
        for i in self.free_dofs:
            R[i] = 0
        # round very small reaction forces to zero
        # tol = 1e-6
        # R.real[abs(R.real) < tol] = 0.0
        return X_red, frequencies, R


def sandwich(B, H, t, material):
    I = 1 / 12 * B * (H ** 3 - (H - 2 * t) ** 3)
    A = 2 * B * t
    return Section(material, A, I, 0)


def main():
    L = 12
    B = 3
    H = 0.5
    t = 0.012
    F = 20000
    q = 1000

    material = Material(E=32e9, rho=1850)
    section = sandwich(B, H, t, material)

    system = System()
    n1 = Node((0, 0))
    b1 = system.add_element(n1, (L / 2, 0), section)
    b2 = system.add_element((L / 2, 0), (L, 0), section)
    n1.add_load(P=(0, -F, 0))
    b1.add_load(q=q)
    b2.add_load(q=q, F=(0, -F, 0), a=0.7)
    b2.add_load(q=q, F=(0, -F, 0), a=0.2)

    # b2 = system.add_element((0, 1), (1, 0), section)
    # system.fix((0, 0))
    # system.load_node((L / 2, 0), Fy=-F)
    b1.n2.add_load((0, -F, 0))
    system.fix(n1)
    system.hinge((L, 0))
    # system.load((1, 0), Fx=0, Fy=-1000)
    X, f, R = system.solve()

    # print(system.constrained_dofs)
    # print(system.reduced_mass_matrix)
    # print(b1.transformed_load_vector)
    print(system.reduced_load_vector)
    print(system.constrained_dofs)
    print(X)
    print(1 / 48 * F * L ** 3 / section.EI)
    print(1 / 384 * q * L ** 4 / section.EI)
    print(q * L + F)
    print(R)
    print(system.free_dofs)
    print(system.constrained_dofs)
    print(system.bounding_box)
    meshed = system.mesh()

    import matplotlib.pyplot as plt

    from fem_plotter import Plotter
    plotter = Plotter(meshed)
    plotter.plot()
    plt.show()


if __name__ == '__main__':
    main()
