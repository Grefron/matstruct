import math

import numpy as np
from matplotlib import pyplot as plt, patches as mpatches


class Plotter:
    # Zorders
    ZCONSTRAINT = 8
    ZELEMENT = 9
    ZNODE = 10

    def __init__(self, system, displacement_factor=1, width=10, height=8):
        self.system = system
        self.figure = plt.figure(figsize=(width, height))
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()
        self.displacement_factor = displacement_factor

        self.setup()

    def clear(self):
        self.ax.clear()

    def setup(self):
        (min_x, min_y), (max_x, max_y) = self.system.bounding_box
        m = 0.1 * self.ref_length
        self.ax.axis('scaled')
        self.ax.axis(xmin=min_x - m, xmax=max_x + m, ymin=min_y - m, ymax=max_y + m)

    @property
    def ref_length(self):
        (min_x, min_y), (max_x, max_y) = self.system.bounding_box
        return max([max_x - min_x, max_y, min_y])

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

    def plot_fixed_supports(self):
        width = height = 0.02 * self.ref_length
        for node in self.system.fixed_nodes:
            support_patch = mpatches.Rectangle((node.x - width * 0.5, - node.y - width * 0.5),
                                               width, height, color='r', zorder=Plotter.ZCONSTRAINT)
            self.ax.add_patch(support_patch)

    def plot_hinged_supports(self):
        radius = 0.007 * self.ref_length
        width = height = 0.02 * self.ref_length
        for node in self.system.hinged_nodes:
            p1 = mpatches.Rectangle((node.x - width * 0.5, - node.y - width * 0.5),
                                    width, height, color='r', zorder=Plotter.ZCONSTRAINT)
            p2 = mpatches.Circle((node.x, node.y), radius=radius, color='w', zorder=Plotter.ZCONSTRAINT)
            self.ax.add_patch(p1)
            self.ax.add_patch(p2)

    def plot_rolled_support(self, max_val):
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

    def plot_nodes(self, show_id=True):
        radius = 0.005 * self.ref_length
        for i, node in enumerate(self.system.nodes):
            p2 = mpatches.Circle((node.x, node.y), radius=radius, color='g', zorder=Plotter.ZNODE)
            self.ax.add_patch(p2)

            if show_id:
                offset = self.ref_length * 0.015
                self.ax.text(node.x + offset, node.y + offset, '%d' % i, color='g', fontsize=10, zorder=Plotter.ZNODE)

    def plot_elements(self, show_id=True):
        for i, e in enumerate(self.system.elements):
            x = e.n1.x, e.n2.x
            y = e.n1.y, e.n2.y
            self.ax.plot(x, y, color='black', zorder=Plotter.ZELEMENT, linewidth=3)

            if show_id:
                # add element ID to plot
                factor = 0.01 * self.ref_length
                x_val = sum(x) / 2 - math.sin(e.alpha) * factor
                y_val = sum(y) / 2 + math.cos(e.alpha) * factor
                self.ax.text(x_val, y_val, "({})".format(i), color='black', fontsize=10, zorder=Plotter.ZELEMENT)

    def plot_structure(self):
        self.plot_elements()
        self.plot_nodes()

    def plot(self):
        self.clear()
        self.setup()
        self.plot_structure()
        self.plot_fixed_supports()
        self.plot_hinged_supports()

    def show(self):
        self.figure.show()

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

    def plot_normal_force(self):
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

    def plot_moment_line(self):
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

    def plot_shear_line(self):
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

    def plot_reaction_forces(self):
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
