import math
import numpy as np
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def is_moving_towards(test_node, node_position, displacement):
    """
    Tests if a node is displacing towards or away from the node.
    :param test_node: Nodes that stands still (vector)
    :param node_position: Node of which the displacment tested (vector)
    :param displacement: (vector)
    :return: (Boolean)
    """
    to_node = test_node - node_position
    return np.dot(to_node, displacement) > 0


def find_nearest(array, value):
    """
    :param array: (numpy array object)
    :param value: (float) value searched for
    :return: (tuple) nearest value, index
    """
    # Subtract the value of the value's in the array. Make the values absolute.
    # The lowest value is the nearest.
    index = (np.abs(array - value)).argmin()
    return array[index], index


class Point:
    def __init__(self, x, z):
        self.x = x
        self.z = z

    def print_values(self):
        print(self.x, self.z)

    def modulus(self):
        return math.sqrt(self.x ** 2 + self.z ** 2)

    def displace_polar(self, alpha, radius, inverse_z_axis=False):
        if inverse_z_axis:
            self.x += math.cos(alpha) * radius
            self.z -= math.sin(alpha) * radius
        else:
            self.x += math.cos(alpha) * radius
            self.z += math.sin(alpha) * radius

    def __add__(self, other):
        x = self.x + other.x
        z = self.z + other.z
        return Point(x, z)

    def __sub__(self, other):
        x = self.x - other.x
        z = self.z - other.z
        return Point(x, z)

    def __truediv__(self, other):
        if type(other) is Point:
            x = self.x / other.x
            z = self.z / other.z
            return Point(x, z)
        else:
            x = self.x / other
            z = self.z / other
            return Point(x, z)

    def __eq__(self, other):
        if self.x == other.x and self.z == other.z:
            return True
        else:
            return False


class Node:
    def __init__(self, ID, Fx=0, Fz=0, Ty=0, ux=0, uz=0, phi_y=0, point=None):
        """
        :param ID: ID of the node, integer
        :param Fx: Value of Fx
        :param Fz: Value of Fz
        :param Ty: Value of Ty
        :param ux: Value of ux
        :param uz: Value of uz
        :param phi_y: Value of phi
        :param point: Point object
        """
        self.ID = ID
        # forces
        self.Fx = Fx
        self.Fz = Fz
        self.Ty = Ty
        # displacements
        self.ux = ux
        self.uz = uz
        self.phi_y = phi_y
        self.point = point

    def show_result(self):
        print("\nID = %s\n"
              "Fx = %s\n"
              "Fz = %s\n"
              "Ty = %s\n"
              "ux = %s\n"
              "uz = %s\n"
              "phi_y = %s" % (self.ID, self.Fx, self.Fz, self.Ty, self.ux, self.uz, self.phi_y))

    def __add__(self, other):
        assert (self.ID == other.ID), "Cannot add nodes as the ID's don't match. The nodes positions don't match."
        Fx = self.Fx + other.Fx
        Fz = self.Fz + other.Fz
        Ty = self.Ty + other.Ty
        ux = self.ux + other.ux
        uz = self.uz + other.uz
        phi_y = self.phi_y + other.phi_y

        return Node(self.ID, Fx, Fz, Ty, ux, uz, phi_y, self.point)

    def __sub__(self, other):
        assert (self.ID == other.ID), "Cannot subtract nodes as the ID's don't match. The nodes positions don't match."
        Fx = self.Fx - other.Fx
        Fz = self.Fz - other.Fz
        Ty = self.Ty - other.Ty
        ux = self.ux - other.ux
        uz = self.uz - other.uz
        phi_y = self.phi_y - other.phi_y

        return Node(self.ID, Fx, Fz, Ty, ux, uz, phi_y, self.point)


class System:
    def __init__(self):
        self.node_ids = []
        self.max_node_id = 2  # minimum is 2, being 1 element
        self.node_objects = []
        self.elements = []
        self.count = 0
        self.system_matrix_locations = []
        self.system_matrix = None
        self.system_force_vector = None
        self.system_displacement_vector = None
        self.shape_system_matrix = None
        self.reduced_force_vector = None
        self.reduced_system_matrix = None
        self.post_processor = SystemLevel(self)
        # list of removed indexes due to conditions
        self.removed_indexes = []
        # list of indexes that remain after conditions are applied
        self.remainder_indexes = []
        # keep track of the nodeID of the supports
        self.supports_fixed = []
        self.supports_hinged = []
        self.supports_roll = []
        self.supports_spring_x = []
        self.supports_spring_z = []
        self.supports_spring_y = []
        self.supports_roll_direction = []
        # keep track of the loads
        self.loads_point = []  # node ids with a point load
        self.loads_q = []  # element ids with a q-load
        self.loads_moment = []
        # results
        self.reaction_forces = []  # node objects

    def add_truss_element(self, coordinates, EA):
        self.add_element(coordinates, EA, 1e-14, type='truss')

    def add_element(self, coordinates, EA, EI, **kwargs):
        """
        :param coordinates: [[x, z], [x, z]]
        :param EA: EA
        :param EI: EI
        :return: Void
        """
        # add the element number
        self.count += 1
        p1 = Point(*coordinates[0])
        p2 = Point(*coordinates[1])

        nodeID1 = 1
        nodeID2 = 2
        existing_node1 = False
        existing_node2 = False

        if len(self.elements) != 0:
            count = 1
            for el in self.elements:
                # check if node 1 of the element meets another node. If so, both have the same nodeID
                if el.point_1 == p1:
                    nodeID1 = el.nodeID1
                    existing_node1 = True
                elif el.point_2 == p1:
                    nodeID1 = el.nodeID2
                    existing_node1 = True
                elif count == len(self.elements) and existing_node1 is False:
                    self.max_node_id += 1
                    nodeID1 = self.max_node_id

                # check for node 2
                if el.point_1 == p2:
                    nodeID2 = el.nodeID1
                    existing_node2 = True
                elif el.point_2 == p2:
                    nodeID2 = el.nodeID2
                    existing_node2 = True
                elif count == len(self.elements) and existing_node2 is False:
                    self.max_node_id += 1
                    nodeID2 = self.max_node_id
                count += 1

        # append the nodes to the system nodes list
        id1 = False
        id2 = False
        for i in self.node_ids:
            if i == nodeID1:
                id1 = True  # nodeID1 allready in system
            if i == nodeID2:
                id2 = True  # nodeID2 allready in system

        if id1 is False:
            self.node_ids.append(nodeID1)
            self.node_objects.append(Node(ID=nodeID1, point=p1))
        if id2 is False:
            self.node_ids.append(nodeID2)
        self.node_objects.append(Node(ID=nodeID2, point=p2))

        # determine the length of the elements
        point = p2 - p1
        l = point.modulus()

        # determine the angle of the element with the global x-axis
        delta_x = p2.x - p1.x
        delta_z = -p2.z - -p1.z  # minus sign to work with an opposite z-axis

        if math.isclose(delta_x, 0, rel_tol=1e-5, abs_tol=1e-9):  # element is vertical
            if delta_z < 0:
                ai = 1.5 * math.pi
            else:
                ai = 0.5 * math.pi
        elif math.isclose(delta_z, 0, rel_tol=1e-5, abs_tol=1e-9):  # element is horizontal
            if delta_x > 0:
                ai = 0
            else:
                ai = math.pi
        elif delta_x > 0 and delta_z > 0:  # quadrant 1 of unity circle
            ai = math.atan(abs(delta_z) / abs(delta_x))
        elif delta_x < 0 < delta_z:  # quadrant 2 of unity circle
            ai = 0.5 * math.pi + math.atan(abs(delta_x) / abs(delta_z))
        elif delta_x < 0 and delta_z < 0:  # quadrant 3 of unity circle
            ai = math.pi + math.atan(abs(delta_z) / abs(delta_x))
        elif delta_z < 0 < delta_x:  # quadrant 4 of unity circle
            ai = 1.5 * math.pi + math.atan(abs(delta_x) / abs(delta_z))
        else:
            raise ValueError("Can't determine the angle of the given element")

        # aj = ai
        # add element
        element = Element(self.count, EA, EI, l, ai, ai, p1, p2)
        element.node_ids.append(nodeID1)
        element.node_ids.append(nodeID2)
        element.nodeID1 = nodeID1
        element.nodeID2 = nodeID2

        element.node_1 = Node(nodeID1)
        element.node_2 = Node(nodeID2)

        element.type = kwargs.get('type', 'general')  # searches key 'type', otherwise set default value 'general'

        self.elements.append(element)

        """
        system matrix [K]

        [fx 1] [K         |  \ node 1 starts at row 1
        |fz 1] |  K       |  /
        |Ty 1] |   K      | /
        |fx 2] |    K     |  \ node 2 starts at row 4
        |fz 2] |     K    |  /
        |Ty 2] |      K   | /
        |fx 3] |       K  |  \ node 3 starts at row 7
        |fz 3] |        K |  /
        [Ty 3] [         K] /

                n   n  n
                o   o  o
                d   d  d
                e   e  e
                1   2  3

        thus with appending numbers in the system matrix: column = row
        """
        # starting row
        # node 1
        row_index_n1 = (element.node_1.ID - 1) * 3

        # node 2
        row_index_n2 = (element.node_2.ID - 1) * 3
        matrix_locations = []

        for _ in range(3):  # ux1, uz1, phi1
            full_row_locations = []
            column_index_n1 = (element.node_1.ID - 1) * 3
            column_index_n2 = (element.node_2.ID - 1) * 3

            for i in range(3):  # matrix row index 1, 2, 3
                full_row_locations.append((row_index_n1, column_index_n1))
                column_index_n1 += 1

            for i in range(3):  # matrix row index 4, 5, 6
                full_row_locations.append((row_index_n1, column_index_n2))
                column_index_n2 += 1

            row_index_n1 += 1
            matrix_locations.append(full_row_locations)

        for _ in range(3):  # ux3, uz3, phi3
            full_row_locations = []
            column_index_n1 = (element.node_1.ID - 1) * 3
            column_index_n2 = (element.node_2.ID - 1) * 3

            for i in range(3):  # matrix row index 1, 2, 3
                full_row_locations.append((row_index_n2, column_index_n1))
                column_index_n1 += 1

            for i in range(3):  # matrix row index 4, 5, 6
                full_row_locations.append((row_index_n2, column_index_n2))
                column_index_n2 += 1

            row_index_n2 += 1
            matrix_locations.append(full_row_locations)
        self.system_matrix_locations.append(matrix_locations)
        return element

    def __assemble_system_matrix(self):
        """
        Shape of the matrix = n nodes * n d.o.f.
        Shape = n * 3
        :return:
        """
        if self.system_matrix is None:
            shape = len(self.node_ids) * 3
            self.shape_system_matrix = shape
            self.system_matrix = np.zeros((shape, shape))

        for i in range(len(self.elements)):

            for row_index in range(len(self.system_matrix_locations[i])):
                count = 0
                for loc in self.system_matrix_locations[i][row_index]:
                    self.system_matrix[loc[0]][loc[1]] += self.elements[i].stiffness_matrix[row_index][count]
                    count += 1

        # returns True if symmetrical.
        return np.allclose((self.system_matrix.transpose()), self.system_matrix)

    def set_force_vector(self, force_list):
        """
        :param force_list: list containing tuples with the
        1. number of the node,
        2. the number of the direction (1 = x, 2 = z, 3 = y)
        3. the force
        [(1, 3, 1000)] node=1, direction=3 (y), force=1000
        list may contain multiple tuples
        :return: Vector with forces on the nodes
        """
        if self.system_force_vector is None:
            self.system_force_vector = np.zeros(self.max_node_id * 3)

        for i in force_list:
            # index = number of the node-1 * nd.o.f. + x, or z, or y
            # (x = 1 * i[1], y = 2 * i[1]
            index = (i[0] - 1) * 3 + i[1] - 1
            # force = i[2]
            self.system_force_vector[index] += i[2]
        return self.system_force_vector

    def set_displacement_vector(self, nodes_list):
        """
        :param nodes_list: list containing tuples with
        1.the node
        2. the d.o.f. that is set
        :return: Vector with the displacements of the nodes (If displacement is not known, the value is set
        to NaN)
        """
        if self.system_displacement_vector is None:
            self.system_displacement_vector = np.empty(self.max_node_id * 3)
            self.system_displacement_vector[:] = np.NaN

        for i in nodes_list:
            index = (i[0] - 1) * 3 + i[1] - 1
            self.system_displacement_vector[index] = 0
        return self.system_displacement_vector

    def __process_conditions(self):
        original_force_vector = np.array(self.system_force_vector)
        original_system_matrix = np.array(self.system_matrix)

        remove_count = 0
        # remove the unsolvable values from the matrix and vectors
        for i in range(self.shape_system_matrix):
            index = i - remove_count
            if self.system_displacement_vector[index] == 0:
                self.system_displacement_vector = np.delete(self.system_displacement_vector, index, 0)
                self.system_force_vector = np.delete(self.system_force_vector, index, 0)
                self.system_matrix = np.delete(self.system_matrix, index, 0)
                self.system_matrix = np.delete(self.system_matrix, index, 1)
                remove_count += 1
                self.removed_indexes.append(i)
            else:
                self.remainder_indexes.append(i)
        self.reduced_force_vector = self.system_force_vector
        self.reduced_system_matrix = self.system_matrix
        self.system_force_vector = original_force_vector
        self.system_matrix = original_system_matrix

    def solve(self):
        assert (self.system_force_vector is not None), "There are no forces on the structure"
        self.__assemble_system_matrix()
        self.__process_conditions()

        # solution of the reduced system (reduced due to support conditions)
        reduced_displacement_vector = np.linalg.solve(self.reduced_system_matrix, self.reduced_force_vector)

        # add the solution of the reduced system in the complete system displacement vector
        self.system_displacement_vector = np.zeros(self.shape_system_matrix)
        count = 0
        for i in self.remainder_indexes:
            self.system_displacement_vector[i] = reduced_displacement_vector[count]
            count += 1

        # determine the displacement vector of the elements
        for el in self.elements:
            index_node_1 = (el.node_1.ID - 1) * 3
            index_node_2 = (el.node_2.ID - 1) * 3

            for i in range(3):  # node 1 ux, uz, phi
                el.element_displacement_vector[i] = self.system_displacement_vector[index_node_1 + i]

            for i in range(3):  # node 2 ux, uz, phi
                el.element_displacement_vector[3 + i] = self.system_displacement_vector[index_node_2 + i]

            el.determine_force_vector()

        # determining the node results in post processing class
        self.post_processor.node_results()
        self.post_processor.reaction_forces()
        self.post_processor.element_results()

        # check the values in the displacement vector for extreme values, indicating a flawed calculation
        for value in np.nditer(self.system_displacement_vector):
            assert (value < 1e6), "The displacements of the structure exceed 1e6. Check your support conditions," \
                                  "or your elements Young's modulus"
        return self.system_displacement_vector

    def hinged_support(self, node):
        """
        adds a hinged support a the given node
        :param node: integer representing the node or node ID
        """
        try:
            node = node.ID
        except AttributeError:
            pass

        self.set_displacement_vector([(node, 1), (node, 2)])

        # add the support to the support list for the plotter
        for obj in self.node_objects:
            if obj.ID == node:
                self.supports_hinged.append(obj)
                break

    def roll_support(self, node, direction=2):
        """
        adds a rollable support at the given node
        :param node: integer representing the nodes ID
        :param direction: integer representing the direction that is fixed: x = 1, z = 2
        """
        try:
            node = node.ID
        except AttributeError:
            pass

        self.set_displacement_vector([(node, direction)])

        # add the support to the support list for the plotter
        for obj in self.node_objects:
            if obj.ID == node:
                self.supports_roll_direction.append(direction)
                self.supports_roll.append(obj)
                break

    def fix_support(self, node):
        """
        adds a fixed support at the given node
        :param node: integer representing the nodes or node ID
        """

        try:
            node = node.ID
        except AttributeError:
            pass

        self.set_displacement_vector([(node, 1), (node, 2), (node, 3)])

        # add the support to the support list for the plotter
        for obj in self.node_objects:
            if obj.ID == node:
                self.supports_fixed.append(obj)
                break

    def spring_support(self, node, translation, K):
        """
        :param translation: Integer representing prevented translation.
        x or 1 = translation in x
        z or 2 = translation in z
        y or 3 or rot = rotation in y
        :param node: Integer representing the nodes ID.
        :param K: Stiffness of the spring

        The stiffness of the spring is added in the system matrix at the location that represents the node and the
        displacement.
        """
        try:
            node = node.ID
        except AttributeError:
            pass

        try:
            translation = {'x': 1, 'z': 2, 'y': 3, 'rot': 3}[translation]
        except KeyError:
            pass

        if self.system_matrix is None:
            shape = len(self.node_ids) * 3
            self.shape_system_matrix = shape
            self.system_matrix = np.zeros((shape, shape))

        # determine the location in the system matrix
        # row and column are the same
        matrix_index = (node - 1) * 3 + translation - 1

        #  first index is row, second is column
        self.system_matrix[matrix_index][matrix_index] += K

        if translation == 1:  # translation spring in x-axis
            self.set_displacement_vector([(node, 2)])

            # add the support to the support list for the plotter
            for obj in self.node_objects:
                if obj.ID == node:
                    self.supports_spring_x.append(obj)
                    break
        elif translation == 2:  # translation spring in z-axis
            self.set_displacement_vector([(node, 1)])

            # add the support to the support list for the plotter
            for obj in self.node_objects:
                if obj.ID == node:
                    self.supports_spring_z.append(obj)
                    break
        elif translation == 3:  # rotational spring in y-axis
            self.set_displacement_vector([(node, 1), (node, 2)])

            # add the support to the support list for the plotter
            for obj in self.node_objects:
                if obj.ID == node:
                    self.supports_spring_y.append(obj)
                    break

    def q_load(self, q, element, direction=1):
        """
        :param element: integer representing the element ID
        :param direction: 1 = towards the element (down-right), -1 = away from the element
        :param q: value of the q-load
        :return:
        """

        try:
            element = element.ID
        except AttributeError:
            pass

        try:
            direction = {'NW':-1, 'SE': 1}[direction]
        except KeyError:
            pass

        self.loads_q.append(element)

        for obj in self.elements:
            if obj.ID == element:
                element = obj
                break

        element.q_load = q * direction

        # determine the left point for the direction of the primary moment
        left_moment = 1 / 12 * q * element.l ** 2 * direction
        right_moment = -1 / 12 * q * element.l ** 2 * direction
        reaction_x = 0.5 * q * element.l * direction * math.sin(element.alpha)
        reaction_z = 0.5 * q * element.l * direction * math.cos(element.alpha)

        # place the primary forces in the 6*6 primary force matrix
        element.element_primary_force_vector[2] += left_moment
        element.element_primary_force_vector[5] += right_moment

        element.element_primary_force_vector[1] -= reaction_z
        element.element_primary_force_vector[4] -= reaction_z
        element.element_primary_force_vector[0] -= reaction_x
        element.element_primary_force_vector[3] -= reaction_x

        # system force vector. System forces = element force * -1
        # first row are the moments
        # second and third row are the reaction forces. The direction
        self.set_force_vector([(element.node_1.ID, 3, -left_moment), (element.node_2.ID, 3, -right_moment),
                               (element.node_1.ID, 2, reaction_z), (element.node_2.ID, 2, reaction_z),
                               (element.node_1.ID, 1, reaction_x), (element.node_2.ID, 1, reaction_x)])

        return self.system_force_vector

    def point_load(self, Fx=0, Fz=0, node=None):
        try:
            node = node.ID
        except AttributeError:
            pass
        self.loads_point.append((node, Fx, Fz))

        if node is not None:
            # system force vector.
            self.set_force_vector([(node, 1, Fx), (node, 2, Fz)])

        return self.system_force_vector

    def moment_load(self, Ty=0, node=None):
        try:
            node = node.ID
        except AttributeError:
            pass
        self.loads_moment.append((node, 3, Ty))

        if node is not None:
            # system force vector.
            self.set_force_vector([(node, 3, Ty)])

    def plot_loads(self):
        return self.plotter.loads()

    def plot_supports(self):
        return self.plotter.supports()

    def plot_structure(self):
        return self.plotter.structure()

    def plot_bending_moment(self):
        return self.plotter.bending_moment()

    def plot_normal_force(self):
        return self.plotter.normal_force()

    def plot_shear_force(self):
        return self.plotter.shear_force()

    def plot_reaction_force(self):
        return self.plotter.reaction_force()

    def plot_displacement(self):
        return self.plotter.displacements()

    def plot(self, structure=False, loads=False, supports=False, bending_moment=False, shear_force=False, normal_force=False,
             reaction_force=False, displacements=False, plot=None, rot=0, center=None):

        if plot is None:
            plot = Plotter(self)
        plot.get_displacement_factor()

        if structure:
            plot.structure()
        if loads:
            plot.loads()
        if supports:
            plot.supports()
        if bending_moment:
            plot.bending_moment()
        if shear_force:
            plot.shear_force()
        if normal_force:
            plot.normal_force()
        if reaction_force:
            plot.reaction_force()
        if displacements:
            plot.displacements(rot=rot, center=center)
        return plot

    def get_node_results_system(self, nodeID=0):
        """
        :param nodeID: (integer) representing the node's ID. If integer = 0, the results of all nodes are returned
        :return:
                if nodeID == 0: (list)
                    Returns a list containing tuples with the results
                if nodeID > 0: (tuple)
                    indexes:
                    0: node's ID
                    1: Fx
                    2: Fz
                    3: Ty
                    4: ux
                    5: uz
                    6: phi_y
        """
        result_list = []
        for obj in self.node_objects:
            if obj.ID == nodeID:
                return obj.ID, obj.Fx, obj.Fz, obj.Ty, obj.ux, obj.uz, obj.phi_y
            else:
                result_list.append((obj.ID, obj.Fx, obj.Fz, obj.Ty, obj.ux, obj.uz, obj.phi_y))
        return result_list

    def get_element_results(self, elementID=0):
        """
        :param elementID: (int) representing the elements ID. If elementID = 0 the results of all elements are returned.
        :return:
                if nodeID == 0: (list)
                    Returns a list containing tuples with the results
                if nodeID > 0: (tuple)
                    indexes:
                    0: elements ID
                    1: length of the element
                    2: elements angle with the global x-axis is radians
                    3: extension
                    4: normal force
                    5: absolute value of maximum deflection
                    6: absolute value of maximum bending moment
                    7: absolute value of maximum shear force
                    8: q-load applying on the element
        """
        result_list = []
        for el in self.elements:
            if elementID == el.ID:
                if el.type == "truss":
                    return (el.ID,
                            el.l,
                            el.alpha,
                            el.extension[0],
                            el.N,
                            None,
                            None,
                            None,
                            None
                            )
                else:
                    return (el.ID,
                            el.l,
                            el.alpha,
                            el.extension[0],
                            el.N,
                            max(abs(min(el.deflection)), abs(max(el.deflection))),
                            max(abs(min(el.bending_moment)), abs(max(el.bending_moment))),
                            max(abs(min(el.shear_force)), abs(max(el.shear_force))),
                            el.q_load
                            )
            else:
                if el.type == "truss":
                    result_list.append(
                        (el.ID,
                         el.l,
                         el.alpha,
                         el.extension[0],
                         el.N,
                         None,
                         None,
                         None,
                         None
                         )
                    )

                else:
                    result_list.append((
                        el.ID,
                        el.l,
                        el.alpha,
                        el.extension[0],
                        el.N,
                        max(abs(min(el.deflection)), abs(max(el.deflection))),
                        max(abs(min(el.bending_moment)), abs(max(el.bending_moment))),
                        max(abs(min(el.shear_force)), abs(max(el.shear_force))),
                        el.q_load
                    )
                    )
        return result_list


class SystemLevel:
    def __init__(self, system):
        self.system = system
        # post processor element level
        self.post_el = ElementLevel(self.system)

    def node_results(self):
        """
        Determines the node results on the system level.
        Results placed in SystemElements class: self.node_objects (list).
        """

        for el in self.system.elements:
            # post processor element level
            self.post_el.node_results(el)

        count = 0
        for node in self.system.node_objects:
            for el in self.system.elements:
                # Minus sign, because the node force is opposite of the element force.
                if el.node_1.ID == node.ID:
                    self.system.node_objects[count] -= el.node_1
                elif el.node_2.ID == node.ID:
                    self.system.node_objects[count] -= el.node_2

            # Loads that are applied on the node of the support. Moment at a hinged support may not lead to reaction
            # moment
            for F_tuple in self.system.loads_moment:
                """
                tuple (nodeID, direction=3, Ty)
                """
                if F_tuple[0] == node.ID:
                    self.system.node_objects[count].Ty += F_tuple[2]

                # The displacements are not summarized. Therefore the displacements are set for every node 1.
                # In order to ensure that every node is overwrote.
                if el.node_1.ID == node.ID:
                    self.system.node_objects[count].ux = el.node_1.ux
                    self.system.node_objects[count].uz = el.node_1.uz
                    self.system.node_objects[count].phi_y = el.node_1.phi_y
                if el.node_2.ID == node.ID:
                    self.system.node_objects[count].ux = el.node_2.ux
                    self.system.node_objects[count].uz = el.node_2.uz
                    self.system.node_objects[count].phi_y = el.node_2.phi_y
            count += 1

    def reaction_forces(self):
        supports = []
        for node in self.system.supports_fixed:
            supports.append(node.ID)
        for node in self.system.supports_hinged:
            supports.append(node.ID)
        for node in self.system.supports_roll:
            supports.append(node.ID)
        for node in self.system.supports_spring_x:
            supports.append(node.ID)
        for node in self.system.supports_spring_z:
            supports.append(node.ID)
        for node in self.system.supports_spring_y:
            supports.append(node.ID)

        for nodeID in supports:
            for node in self.system.node_objects:
                if nodeID == node.ID:
                    node = copy.copy(node)
                    node.Fx *= -1
                    node.Fz *= -1
                    node.Ty *= -1
                    node.ux = None
                    node.uz = None
                    node.phi_y = None
                    self.system.reaction_forces.append(node)

    def element_results(self):
        """
        Determines the element results for al elements in the system on element level.
        """
        for el in self.system.elements:
            con = 100
            self.post_el.determine_bending_moment(el, con)
            self.post_el.determine_shear_force(el, con)
            self.post_el.determine_displacements(el, con)


class ElementLevel:
    def __init__(self, system):
        self.system = system

    def node_results(self, element):
        """
        Determine node results on the element level.
        """
        element.node_1 = Node(
            ID=element.node_ids[0],
            Fx=element.element_force_vector[0] + element.element_primary_force_vector[0],
            Fz=element.element_force_vector[1] + element.element_primary_force_vector[1],
            Ty=element.element_force_vector[2] + element.element_primary_force_vector[2],
            ux=element.element_displacement_vector[0],
            uz=element.element_displacement_vector[1],
            phi_y=element.element_displacement_vector[2],
        )

        element.node_2 = Node(
            ID=element.node_ids[1],
            Fx=element.element_force_vector[3] + element.element_primary_force_vector[3],
            Fz=element.element_force_vector[4] + element.element_primary_force_vector[4],
            Ty=element.element_force_vector[5] + element.element_primary_force_vector[5],
            ux=element.element_displacement_vector[3],
            uz=element.element_displacement_vector[4],
            phi_y=element.element_displacement_vector[5]
        )
        self._determine_normal_force(element)

    @staticmethod
    def _determine_normal_force(element):
        test_node = np.array([element.point_1.x, element.point_1.z])
        node_position = np.array([element.point_2.x, element.point_2.z])
        displacement = np.array([element.node_2.ux - element.node_1.ux, element.node_2.uz - element.node_1.uz])

        force_towards = is_moving_towards(test_node, node_position, displacement)
        N = abs(math.sin(element.alpha) * element.node_1.Fz) + abs(math.cos(element.alpha) * element.node_1.Fx)

        if force_towards:
            element.N = -N
        else:
            element.N = N

    @staticmethod
    def determine_bending_moment(element, con):
        dT = -(element.node_2.Ty + element.node_1.Ty)  # T2 - (-T1)

        x_val = np.linspace(0, 1, con)
        m_val = np.empty(con)

        # determine moment for 0 < x < length of the element
        count = 0
        for i in x_val:
            x = i * element.l
            x_val[count] = x
            m_val[count] = element.node_1.Ty + i * dT

            if element.q_load:
                q_part = (-0.5 * -element.q_load * x ** 2 + 0.5 * -element.q_load * element.l * x)
                m_val[count] += q_part
            count += 1
        element.bending_moment = m_val

    @staticmethod
    def determine_shear_force(element, con):
        """
        Determines the shear force by differentiating the bending moment.
        :param element: (object) of the Element class
        """
        dV = np.diff(element.bending_moment)
        dx = element.l / (con - 1)
        shear_force = dV / dx

        # Due to differentiation the first and the last values must be corrected.
        correction = shear_force[1] - shear_force[0]
        shear_force = np.insert(shear_force, 0, [shear_force[0] - 0.5 * correction])
        shear_force = np.insert(shear_force, con, [shear_force[-1] + 0.5 * correction])
        element.shear_force = shear_force

    @staticmethod
    def determine_displacements(element, con):
        """
        Determines the displacement by integrating the bending moment.
        :param element: (object) of the Element class

        deflection w =
        EI * (d^4/dx^4 * w(x)) = q

        solution of differential equation:

        w = 1/24 qx^4 + 1/6 c1x^3 + 1/2 c2 x^2 + c3x + c4  ---> c4 = w(0)
        phi = -1/6 qx^3/EI - 1/2c1x^2 -c2x -c3  ---> c3 = -phi
        M = EI(-1/2qx^2/EI -c1x -c2)  ---> c2 = -M/EI
        V = EI(-qx/EI -c1)  ---> c1 = -V/EI
        """
        if element.type == 'general':
            c1 = -element.shear_force[0] / element.EI
            c2 = -element.bending_moment[0] / element.EI
            c3 = element.node_1.phi_y
            c4 = 0
            w = np.empty(con)
            dx = element.l / con

            for i in range(con):
                x = (i + 1) * dx
                w[i] = 1 / 6 * c1 * x ** 3 + 0.5 * c2 * x ** 2 + c3 * x + c4
                if element.q_load:
                    w[i] += 1 / 24 * -element.q_load * x ** 4 / element.EI
            element.deflection = -w

            # max deflection
            element.max_deflection = max(abs(min(w)), abs(max(w)))

        """
        Extension
        """

        u = element.N / element.EA * element.l
        du = u / con

        element.extension = np.empty(con)

        for i in range(con):
            u = du * (i + 1)
            element.extension[i] = u


"""
The matrices underneath are for slender beams, where the most deformation occurs due to bending.
Shear deformation is not taken into account.
"""


class Element:
    def __init__(self, ID, EA, EI, l, ai, aj, point_1, point_2):
        """
        :param ID: integer representing the elements ID
        :param EA: Young's modulus * Area
        :param EI: Young's modulus * Moment of Inertia
        :param l: length
        :param ai: angle between element and x-axis
        :param aj: = ai
        :param point_1: point object
        :param point_2: point object
        """
        self.ID = ID
        self.type = None
        self.EA = EA
        self.EI = EI
        self.l = l
        self.point_1 = point_1
        self.point_2 = point_2
        self.alpha = ai
        self.kinematic_matrix = kinematic_matrix(ai, aj, l)
        self.constitutive_matrix = constitutive_matrix(EA, EI, l)
        self.stiffness_matrix = stiffness_matrix(self.constitutive_matrix, self.kinematic_matrix)
        self.nodeID1 = None
        self.nodeID2 = None
        self.node_ids = []
        self.node_1 = None
        self.node_2 = None
        self.element_displacement_vector = np.empty(6)
        self.element_primary_force_vector = np.zeros(6)
        self.element_force_vector = None
        self.q_load = None
        self.N = None
        self.bending_moment = None
        self.shear_force = None
        self.deflection = None
        self.extension = None
        self.max_deflection = None

    def determine_force_vector(self):
        self.element_force_vector = np.dot(self.stiffness_matrix, self.element_displacement_vector)
        return self.element_force_vector


def kinematic_matrix(ai, aj, l):
    return np.array([[-math.cos(ai), math.sin(ai), 0, math.cos(aj), -math.sin(aj), 0],
                     [math.sin(ai) / l, math.cos(ai) / l, -1, -math.sin(aj) / l, -math.cos(aj) / l, 0],
                     [-math.sin(ai) / l, -math.cos(ai) / l, 0, math.sin(aj) / l, math.cos(aj) / l, 1]])


def constitutive_matrix(EA, EI, l):
    return np.array([[EA / l, 0, 0],
                     [0, 4 * EI / l, -2 * EI / l],
                     [0, -2 * EI / l, 4 * EI / l]])


def stiffness_matrix(var_constitutive_matrix, var_kinematic_matrix):
    kinematic_transposed_times_constitutive = np.dot(var_kinematic_matrix.transpose(), var_constitutive_matrix)
    return np.dot(kinematic_transposed_times_constitutive, var_kinematic_matrix)


class Plotter:
    def __init__(self, system):
        self.system = system
        self.max_val = None
        self.max_force = 0
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        # plt.tight_layout()
        self.figure.tight_layout()
        self.displacement_factor = None

    def clear(self):
        self.ax.clear()

    def __set_factor(self, value_1, value_2):
        """
        :param value_1: value of the force/ moment at point 1
        :param value_2: value of the force/ moment at point 2
        :return: factor for scaling the force/moment in the plot
        """

        if abs(value_1) > self.max_force:
            self.max_force = abs(value_1)
        if abs(value_2) > self.max_force:
            self.max_force = abs(value_2)

        if math.isclose(self.max_force, 0):
            factor = 0.1
        else:
            factor = 0.15 * self.max_val / self.max_force
        return factor

    def __fixed_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """

        width = 0.05 * max_val
        height = 0.05 * max_val
        for node in self.system.supports_fixed:
            support_patch = mpatches.Rectangle((node.point.x - width * 0.5, - node.point.z - width * 0.5),
                                               width, height, color='r', zorder=9)
            self.ax.add_patch(support_patch)

    def __hinged_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        radius = 0.04 * max_val
        for node in self.system.supports_hinged:
            support_patch = mpatches.RegularPolygon((node.point.x, -node.point.z - radius),
                                                    numVertices=3, radius=radius, color='r', zorder=9)
            self.ax.add_patch(support_patch)

    def __roll_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        radius = 0.03 * max_val
        count = 0
        for node in self.system.supports_roll:

            direction = self.system.supports_roll_direction[count]

            if direction == 2:  # horizontal roll
                support_patch = mpatches.RegularPolygon((node.point.x, -node.point.z - radius),
                                                        numVertices=3, radius=radius, color='r', zorder=9)
                self.ax.add_patch(support_patch)
                y = -node.point.z - 2 * radius
                self.ax.plot([node.point.x - radius, node.point.x + radius], [y, y], color='r')
            elif direction == 1:  # vertical roll
                center = 0
                x1 = center + math.cos(math.pi) * radius + node.point.x + radius
                z1 = center + math.sin(math.pi) * radius - node.point.z
                x2 = center + math.cos(math.radians(90)) * radius + node.point.x + radius
                z2 = center + math.sin(math.radians(90)) * radius - node.point.z
                x3 = center + math.cos(math.radians(270)) * radius + node.point.x + radius
                z3 = center + math.sin(math.radians(270)) * radius - node.point.z

                triangle = np.array([[x1, z1], [x2, z2], [x3, z3]])
                # translate the support to the node

                support_patch = mpatches.Polygon(triangle, color='r', zorder=9)
                self.ax.add_patch(support_patch)

                y = -node.point.z - radius
                self.ax.plot([node.point.x + radius * 1.5, node.point.x + radius * 1.5], [y, y + 2 * radius],
                             color='r')
            count += 1

    def __rotating_spring_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        radius = 0.04 * max_val

        for node in self.system.supports_spring_y:
            r = np.arange(0, radius, 0.001)
            theta = 25 * math.pi * r / (0.2 * max_val)
            x_val = []
            y_val = []

            count = 0
            for angle in theta:
                x = math.cos(angle) * r[count] + node.point.x
                y = math.sin(angle) * r[count] - radius - node.point.z
                x_val.append(x)
                y_val.append(y)
                count += 1

            self.ax.plot(x_val, y_val, color='r', zorder=9)

            # Triangle
            support_patch = mpatches.RegularPolygon((node.point.x, -node.point.z - radius * 3),
                                                    numVertices=3, radius=radius * 0.9, color='r', zorder=9)
            self.ax.add_patch(support_patch)

    def __spring_support_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        h = 0.04 * max_val
        left = -0.5 * h
        right = 0.5 * h
        dh = 0.2 * h

        for node in self.system.supports_spring_z:
            yval = np.arange(0, -9, -1)
            yval = yval * dh
            xval = np.array([0, 0, left, right, left, right, left, 0, 0])

            yval = yval - node.point.z
            xval = xval + node.point.x
            # Triangle
            support_patch = mpatches.RegularPolygon((node.point.x, -node.point.z - h * 2.6),
                                                    numVertices=3, radius=h * 0.9, color='r', zorder=10)

            self.ax.plot(xval, yval, color='r', zorder=10)
            self.ax.add_patch(support_patch)

        for node in self.system.supports_spring_x:
            xval = np.arange(0, 9, 1)
            xval *= dh
            yval = np.array([0, 0, left, right, left, right, left, 0, 0])

            xval += node.point.x
            yval -= node.point.z
            # Triangle
            support_patch = mpatches.RegularPolygon((node.point.x + h * 1.7, -node.point.z - h),
                                                    numVertices=3, radius=h * 0.9, color='r', zorder=10)

            self.ax.plot(xval, yval, color='r', zorder=10)
            self.ax.add_patch(support_patch)

    def __q_load_patch(self, max_val):
        """
        :param max_val: max scale of the plot

        xn1;yn1  q-load   xn1;yn1
        -------------------
        |__________________|
        x1;y1  element    x2;y2
        """
        h = 0.05 * max_val

        for q_ID in self.system.loads_q:
            for el in self.system.elements:
                if el.ID == q_ID:
                    if el.q_load > 0:
                        direction = 1
                    else:
                        direction = -1

                    x1 = el.point_1.x
                    y1 = -el.point_1.z
                    x2 = el.point_2.x
                    y2 = -el.point_2.z
                    # - value, because the positive z of the system is opposite of positive y of the plotter
                    xn1 = x1 + math.sin(-el.alpha) * h * direction
                    yn1 = y1 + math.cos(-el.alpha) * h * direction
                    xn2 = x2 + math.sin(-el.alpha) * h * direction
                    yn2 = y2 + math.cos(-el.alpha) * h * direction
                    self.ax.plot([x1, xn1, xn2, x2], [y1, yn1, yn2, y2], color='g')

                    # arrow
                    xa_1 = (x2 - x1) * 0.2 + x1 + math.sin(-el.alpha) * 0.8 * h * direction
                    ya_1 = (y2 - y1) * 0.2 + y1 + math.cos(-el.alpha) * 0.8 * h * direction
                    len_x = math.sin(-el.alpha - math.pi) * 0.6 * h * direction
                    len_y = math.cos(-el.alpha - math.pi) * 0.6 * h * direction
                    xt = xa_1 + math.sin(-el.alpha) * 0.4 * h * direction
                    yt = ya_1 + math.cos(-el.alpha) * 0.4 * h * direction
                    # fc = face color, ec = edge color
                    self.ax.arrow(xa_1, ya_1, len_x, len_y, head_width=h * 0.25, head_length=0.2 * h, ec='g',
                                  fc='g')
                    self.ax.text(xt, yt, "q=%d" % el.q_load, color='k', fontsize=9, zorder=10)

    def __arrow_patch_values(self, Fx, Fz, node, h):
        """
        :param Fx: (float)
        :param Fz: (float)

        -- One of the above must be zero. The function created to find the non-zero F-direction.

        :param node: (Node object)
        :param h: (float) Is a scale variable
        :return: Variables for the matplotlib plotter
        """
        if Fx > 0:  # Fx is positive
            x = node.point.x - h
            y = -node.point.z
            len_x = 0.8 * h
            len_y = 0
            F = Fx
        elif Fx < 0:  # Fx is negative
            x = node.point.x + h
            y = -node.point.z
            len_x = -0.8 * h
            len_y = 0
            F = Fx
        elif Fz > 0:  # Fz is positive
            x = node.point.x
            y = -node.point.z + h
            len_x = 0
            len_y = -0.8 * h
            F = Fz
        else:  # Fz is negative
            x = node.point.x
            y = -node.point.z - h
            len_x = 0
            len_y = 0.8 * h
            F = Fz

        return x, y, len_x, len_y, F

    def __point_load_patch(self, max_val):
        """
        :param max_val: max scale of the plot
        """
        h = 0.1 * max_val

        for F_tuple in self.system.loads_point:
            for node in self.system.node_objects:
                if node.ID == F_tuple[0]:  # F_tuple[0] = ID
                    sol = self.__arrow_patch_values(F_tuple[1], F_tuple[2], node, h)
                    x = sol[0]
                    y = sol[1]
                    len_x = sol[2]
                    len_y = sol[3]
                    F = sol[4]

                    self.ax.arrow(x, y, len_x, len_y, head_width=h * 0.15, head_length=0.2 * h, ec='b',
                                  fc='orange',
                                  zorder=11)
                    self.ax.text(x, y, "F=%d" % F, color='k', fontsize=9, zorder=10)

    def __moment_load_patch(self, max_val):

        h = 0.2 * max_val

        for F_tuple in self.system.loads_moment:
            for node in self.system.node_objects:
                if node.ID == F_tuple[0]:
                    if F_tuple[2] > 0:
                        self.ax.plot(node.point.x, -node.point.z, marker=r'$\circlearrowleft$', ms=25,
                                     color='orange')
                    else:
                        self.ax.plot(node.point.x, -node.point.z, marker=r'$\circlearrowright$', ms=25,
                                     color='orange')
                    self.ax.text(node.point.x + h * 0.2, -node.point.z + h * 0.2, "T=%d" % F_tuple[2], color='k',
                                 fontsize=9, zorder=10)

    def supports(self):
        _, _, max_val = self.get_max_val()

        self.__fixed_support_patch(max_val)
        self.__hinged_support_patch(max_val)
        self.__roll_support_patch(max_val)
        self.__rotating_spring_support_patch(max_val)
        self.__spring_support_patch(max_val)

        return self.figure

    def loads(self):
        _, _, max_val = self.get_max_val()

        self.__q_load_patch(max_val)
        self.__point_load_patch(max_val)
        self.__moment_load_patch(max_val)

        return self.figure

    def structure(self):
        """
        :param loads: (boolean) if True, loads are plotted.
        :param supports: (boolean) if True, supports are plotted.
        :return:
        """
        center_x, center_z, max_val = self.get_max_val()

        self.max_val = max_val
        offset = max_val
        plusxrange = center_x + offset
        plusyrange = center_z + offset
        minxrange = center_x - offset
        minyrange = center_z - offset

        self.ax.axis([minxrange, plusxrange, minyrange, plusyrange])

        for el in self.system.elements:
            axis_values = plot_values_element(el)
            x_val = axis_values[0]
            y_val = axis_values[1]

            # add node ID to plot
            offset = max_val * 0.015
            self.ax.text(x_val[0] + offset, y_val[0] + offset, '%d' % el.nodeID1, color='g', fontsize=9, zorder=10)
            self.ax.text(x_val[-1] + offset, y_val[-1] + offset, '%d' % el.nodeID2, color='g', fontsize=9,
                         zorder=10)

            # add element ID to plot
            factor = 0.02 * self.max_val
            x_val = (x_val[0] + x_val[-1]) / 2 - math.sin(el.alpha) * factor
            y_val = (y_val[0] + y_val[-1]) / 2 + math.cos(el.alpha) * factor

            self.ax.text(x_val, y_val, "%d" % el.ID, color='r', fontsize=9, zorder=10)

        return self.figure

    def get_max_val(self):
        max_x = 0
        max_z = 0
        min_x = 0
        min_z = 0
        for el in self.system.elements:
            # plot structure
            axis_values = plot_values_element(el)
            x_val = axis_values[0]
            y_val = axis_values[1]
            self.ax.plot(x_val, y_val, color='black', marker='s')

            # determine max values for scaling the plotter
            max_x = max([abs(x) for x in x_val]) if max([abs(x) for x in x_val]) > max_x else max_x
            max_z = max([abs(x) for x in y_val]) if max([abs(x) for x in y_val]) > max_z else max_z

            min_x = min([abs(x) for x in x_val]) if min([abs(x) for x in x_val]) < min_x else min_x
            min_z = min([abs(x) for x in y_val]) if min([abs(x) for x in y_val]) < min_z else min_z
            center_x = (max_x - min_x) / 2 + min_x
            center_z = (max_z - min_z) / 2 + min_x
        max_val = max(max_x, max_z)
        return center_x, center_z, max_val

    def _add_node_values(self, x_val, y_val, value_1, value_2, digits):
        # add value to plot
        self.ax.text(x_val[1] - 2 / self.max_val, y_val[1] + 2 / self.max_val, "%s" % round(value_1, digits),
                     fontsize=9, ha='center', va='center', )
        self.ax.text(x_val[-2] - 2 / self.max_val, y_val[-2] + 2 / self.max_val, "%s" % round(value_2, digits),
                     fontsize=9, ha='center', va='center', )

    def _add_element_values(self, x_val, y_val, value, index, digits=2):
        self.ax.text(x_val[index], y_val[index], "%s" % round(value, digits),
                     fontsize=9, ha='center', va='center', )

    def plot_result(self, axis_values, force_1=None, force_2=None, digits=2, node_results=True, rot=0, center=None):
        # plot force
        x_val = axis_values[0]
        y_val = axis_values[1]

        # Transform
        if center is not None:
            cx, cy = center
        else:
            cx, cy = 0, 0
        plot_x_val = (x_val - cx) * math.cos(rot) + (y_val - cy) * math.sin(rot) + cx
        plot_y_val = (x_val - cx) * math.sin(rot) + (y_val - cy) * math.cos(rot) + cy
        self.ax.plot(plot_x_val, plot_y_val, color='b')

        if node_results:
            self._add_node_values(x_val, y_val, force_1, force_2, digits)

    def normal_force(self):
        self.max_force = 0

        # determine max factor for scaling
        for el in self.system.elements:
            factor = self.__set_factor(el.N, el.N)

        for el in self.system.elements:
            if math.isclose(el.N, 0, rel_tol=1e-5, abs_tol=1e-9):
                pass
            else:
                axis_values = plot_values_normal_force(el, factor)
                self.plot_result(axis_values, el.N, el.N)

                point = (el.point_2 - el.point_1) / 2 + el.point_1
                if el.N < 0:
                    point.displace_polar(alpha=el.alpha + 0.5 * math.pi, radius=0.5 * el.N * factor,
                                         inverse_z_axis=True)

                    self.ax.text(point.x, -point.z, "-", ha='center', va='center',
                                 fontsize=20, color='b')
                if el.N > 0:
                    point.displace_polar(alpha=el.alpha + 0.5 * math.pi, radius=0.5 * el.N * factor,
                                         inverse_z_axis=True)

                    self.ax.text(point.x, -point.z, "+", ha='center', va='center',
                                 fontsize=14, color='b')
        return self.figure

    def bending_moment(self):
        self.max_force = 0
        con = len(self.system.elements[0].bending_moment)

        # determine max factor for scaling
        factor = 0
        for el in self.system.elements:
            if el.q_load:
                m_sag = (el.node_1.Ty - el.node_2.Ty) * 0.5 - 1 / 8 * el.q_load * el.l ** 2
                value_1 = max(abs(el.node_1.Ty), abs(m_sag))
                value_2 = max(value_1, abs(el.node_2.Ty))
                factor = self.__set_factor(value_1, value_2)
            else:
                factor = self.__set_factor(el.node_1.Ty, el.node_2.Ty)

        # determine the axis values
        for el in self.system.elements:
            if math.isclose(el.node_1.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and \
                    math.isclose(el.node_2.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and el.q_load is None:
                # If True there is no bending moment, so no need for plotting.
                pass

            else:
                axis_values = plot_values_bending_moment(el, factor, con)
                self.plot_result(axis_values, abs(el.node_1.Ty), abs(el.node_2.Ty))

                if el.q_load:
                    m_sag = min(el.bending_moment)
                    index = find_nearest(el.bending_moment, m_sag)[1]
                    offset = -self.max_val * 0.05
                    x = axis_values[0][index] + math.sin(-el.alpha) * offset
                    y = axis_values[1][index] + math.cos(-el.alpha) * offset
                    self.ax.text(x, y, "%s" % round(m_sag, 1),
                                 fontsize=9)
        return self.figure

    def shear_force(self):
        self.max_force = 0

        # determine max factor for scaling
        for el in self.system.elements:
            shear_1 = max(el.shear_force)
            shear_2 = min(el.shear_force)
            factor = self.__set_factor(shear_1, shear_2)

        for el in self.system.elements:
            if math.isclose(el.node_1.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and \
                    math.isclose(el.node_2.Ty, 0, rel_tol=1e-5, abs_tol=1e-9) and el.q_load is None:
                # If True there is no bending moment, thus no shear force, so no need for plotting.
                pass
            else:
                axis_values = plot_values_shear_force(el, factor)
                shear_1 = axis_values[-2]
                shear_2 = axis_values[-1]
                self.plot_result(axis_values, shear_1, shear_2)
        return self.figure

    def reaction_force(self):
        _, _, max_val = self.get_max_val()
        self.max_val = max_val
        h = 0.2 * self.max_val
        max_force = 0

        for node in self.system.reaction_forces:
            max_force = abs(node.Fx) if abs(node.Fx) > max_force else max_force
            max_force = abs(node.Fz) if abs(node.Fz) > max_force else max_force

        for node in self.system.reaction_forces:
            if not math.isclose(node.Fx, 0, rel_tol=1e-5, abs_tol=1e-9):
                # x direction
                scale = abs(node.Fx) / max_force * h
                sol = self.__arrow_patch_values(node.Fx, 0, node, scale)
                x = sol[0]
                y = sol[1]
                len_x = sol[2]
                len_y = sol[3]

                self.ax.arrow(x, y, len_x, len_y, head_width=h * 0.15, head_length=0.2 * scale, ec='b',
                              fc='orange',
                              zorder=11)
                self.ax.text(x, y, "R=%s" % round(node.Fx, 2), color='k', fontsize=9, zorder=10)

            if not math.isclose(node.Fz, 0, rel_tol=1e-5, abs_tol=1e-9):
                # z direction
                scale = abs(node.Fz) / max_force * h
                sol = self.__arrow_patch_values(0, node.Fz, node, scale)
                x = sol[0]
                y = sol[1]
                len_x = sol[2]
                len_y = sol[3]

                self.ax.arrow(x, y, len_x, len_y, head_width=h * 0.15, head_length=0.2 * scale, ec='b',
                              fc='orange',
                              zorder=11)
                self.ax.text(x, y, "R=%s" % round(node.Fz, 2), color='k', fontsize=9, zorder=10)

            if not math.isclose(node.Ty, 0, rel_tol=1e-5, abs_tol=1e-9):
                """
                'r: regex
                '$...$': render the strings using mathtext
                """
                if node.Ty > 0:
                    self.ax.plot(node.point.x, -node.point.z, marker=r'$\circlearrowleft$', ms=25,
                                 color='orange')
                if node.Ty < 0:
                    self.ax.plot(node.point.x, -node.point.z, marker=r'$\circlearrowright$', ms=25,
                                 color='orange')

                self.ax.text(node.point.x + h * 0.2, -node.point.z + h * 0.2, "T=%s" % round(node.Ty, 2),
                             color='k', fontsize=9, zorder=10)
        return self.figure

    def get_displacement_factor(self):
        self.max_force = 0
        center_x, center_z, max_val = self.get_max_val()
        self.max_val = max_val
        # determine max factor for scaling
        for el in self.system.elements:
            u_node = max(abs(el.node_1.ux), abs(el.node_1.uz))
            if el.type == "general":
                factor = self.__set_factor(el.max_deflection, u_node)
            else:  # element is truss
                factor = self.__set_factor(u_node, 0)
        return factor

    @property
    def displacement_factor(self):
        if self._displacement_factor is None:
            return self.get_displacement_factor()
        else:
            return self._displacement_factor

    @displacement_factor.setter
    def displacement_factor(self, value):
        self._displacement_factor = value

    def displacements(self, rot=0, center=None):
        factor = self.displacement_factor

        for el in self.system.elements:
            axis_values = plot_values_deflection(el, factor)
            self.plot_result(axis_values, node_results=False, rot=rot, center=center)

            if el.type == "general":
                # index of the max deflection
                if max(abs(el.deflection)) > min(abs(el.deflection)):
                    index = el.deflection.argmax()
                else:
                    index = el.deflection.argmin()
                self._add_element_values(axis_values[0], axis_values[1], el.deflection[index], index, 3)


"""
### element functions ###
"""


def plot_values_element(element):
    x_val = [element.point_1.x, element.point_2.x]
    y_val = [-element.point_1.z, -element.point_2.z]
    return x_val, y_val


def plot_values_shear_force(element, factor=1):
    x1 = element.point_1.x
    y1 = -element.point_1.z
    x2 = element.point_2.x
    y2 = -element.point_2.z

    shear_1 = element.shear_force[0]
    shear_2 = element.shear_force[-1]

    # apply angle ai
    x_1 = x1 + shear_1 * math.sin(-element.alpha) * factor
    y_1 = y1 + shear_1 * math.cos(-element.alpha) * factor
    x_2 = x2 + shear_2 * math.sin(-element.alpha) * factor
    y_2 = y2 + shear_2 * math.cos(-element.alpha) * factor

    x_val = np.array([x1, x_1, x_2, x2])
    y_val = np.array([y1, y_1, y_2, y2])
    return x_val, y_val, shear_1, shear_2


def plot_values_normal_force(element, factor):
    x1 = element.point_1.x
    y1 = -element.point_1.z
    x2 = element.point_2.x
    y2 = -element.point_2.z

    x_1 = x1 + element.N * math.cos(0.5 * math.pi + element.alpha) * factor
    y_1 = y1 + element.N * math.sin(0.5 * math.pi + element.alpha) * factor
    x_2 = x2 + element.N * math.cos(0.5 * math.pi + element.alpha) * factor
    y_2 = y2 + element.N * math.sin(0.5 * math.pi + element.alpha) * factor

    x_val = [x1, x_1, x_2, x2]
    y_val = [y1, y_1, y_2, y2]
    return x_val, y_val


def plot_values_bending_moment(element, factor, con):
    """
    :param element: (object) of the Element class
    :param factor: (float) scaling the plot
    :param con: (integer) amount of x-values
    :return:
    """
    x1 = element.point_1.x
    y1 = -element.point_1.z
    x2 = element.point_2.x
    y2 = -element.point_2.z

    # Determine forces for horizontal element ai = 0
    T_left = element.node_1.Ty
    T_right = -element.node_2.Ty

    # apply angle ai
    x1 += T_left * math.sin(-element.alpha) * factor
    y1 += T_left * math.cos(-element.alpha) * factor
    x2 += T_right * math.sin(-element.alpha) * factor
    y2 += T_right * math.cos(-element.alpha) * factor

    x_val = np.linspace(0, 1, con)
    y_val = np.empty(con)
    dx = x2 - x1
    dy = y2 - y1

    # determine moment for 0 < x < length of the element
    count = 0
    for i in x_val:
        x_val[count] = x1 + i * dx
        y_val[count] = y1 + i * dy

        if element.q_load:
            x = i * element.l
            q_part = (-0.5 * -element.q_load * x ** 2 + 0.5 * -element.q_load * element.l * x)

            x_val[count] += math.sin(-element.alpha) * q_part * factor
            y_val[count] += math.cos(-element.alpha) * q_part * factor
        count += 1

    x_val = np.append(x_val, element.point_2.x)
    y_val = np.append(y_val, -element.point_2.z)
    x_val = np.insert(x_val, 0, element.point_1.x)
    y_val = np.insert(y_val, 0, -element.point_1.z)

    return x_val, y_val


def plot_values_deflection(element, factor):
    ux1 = element.node_1.ux * factor
    uz1 = -element.node_1.uz * factor
    ux2 = element.node_2.ux * factor
    uz2 = -element.node_2.uz * factor

    x1 = element.point_1.x + ux1
    y1 = -element.point_1.z + uz1

    if element.type == "general":
        n_val = len(element.deflection)
        x_val = np.empty(n_val)
        y_val = np.empty(n_val)

        dl = element.l / n_val

        for i in range(n_val):
            dx = (element.deflection[i] * math.sin(element.alpha)
                  + element.extension[i] * math.cos(element.alpha)) * factor
            dy = (element.deflection[i] * -math.cos(element.alpha)
                  + element.extension[i] * math.sin(element.alpha)) * factor

            x = (i + 1) * dl * math.cos(element.alpha)
            y = (i + 1) * dl * math.sin(element.alpha)

            x_val[i] = x1 + x + dx
            y_val[i] = y1 + y + dy

        x_val = np.insert(x_val, 0, x1)
        y_val = np.insert(y_val, 0, y1)

    else:  # truss element has no bending
        x2 = element.point_2.x + ux2
        y2 = -element.point_2.z + uz2
        x_val = np.array([x1, x2])
        y_val = np.array([y1, y2])

    return x_val, y_val
