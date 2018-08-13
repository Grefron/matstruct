from shapely import geometry as geom


class Polygon(geom.Polygon):
    def __init__(self, material, *args, shear_factor=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.material = material
        self.shear_factor = shear_factor

    @property
    def A(self):
        return self.area

    @property
    def As(self):
        return self.shear_factor * self.area

    @property
    def Ex(self):
        try:
            return self.material.Ex
        except AttributeError:
            return self.material.E

    @property
    def Ey(self):
        try:
            return self.material.Ey
        except AttributeError:
            return self.material.E

    @property
    def Gxy(self):
        try:
            return self.material.Gxy
        except AttributeError:
            return self.material.G

    @property
    def EAx(self):
        return self.Ex * self.area

    @property
    def EAy(self):
        return self.Ey * self.area

    @property
    def GA(self):
        return self.shear_factor * self.Gxy * self.area

    def _cw_points(self):
        """
        Iterate through points, clockwise
        :return: 
        """
        cw = -1
        p = geom.polygon.orient(self, cw)
        for p in p.exterior.coords:
            yield p

    def _ccw_points(self):
        """
        Iterate through points, counterclockwise
        :return: 
        """
        ccw = 1
        p = geom.polygon.orient(self, ccw)
        for p in p.exterior.coords:
            yield p

    def lines(self):
        n = len(self.exterior.coords) - 1  # number of sides
        l = []
        for i in range(n):
            l.append(geom.LineString(self.exterior.coords[i], self.exterior.coords[i + 1]))
        return l

    def lines_in_range(self, x_range=None, y_range=None):
        if x_range is not None:
            xmin, xmax = x_range
        if y_range is not None:
            ymin, ymax = y_range

        lines = []
        for l in self.lines():
            x1, y1 = l.coords[0]
            x2, y2 = l.coords[1]
            fit = True
            if x_range is not None:
                if not (xmin < x1 < xmax and xmin < x2 < xmax):
                    fit = False
            if y_range is not None:
                if not (ymin < y1 < ymax and ymin < y2 < ymax):
                    fit = False
            if fit:
                lines.append(l)
        return lines

    @property
    def first_moment(self):
        S_x = S_y = 0
        x2, y2 = list(self.exterior.coords)[0]
        for x1, y1 in self._cw_points():
            f = x1 * y2 - x2 * y1
            S_y += (x1 + x2) * f / 6
            S_x += (y1 + y2) * f / 6
            x2, y2 = x1, y1
        return S_x, S_y

    @property
    def second_moment(self):
        I_x = I_y = 0
        x2, y2 = list(self.exterior.coords)[0]
        for point in self._cw_points():
            x1, y1 = point
            f = x1 * y2 - x2 * y1
            I_x += (y1 ** 2 + y1 * y2 + y2 ** 2) * f / 12
            I_y += (x1 ** 2 + x1 * x2 + x2 ** 2) * f / 12
            x2, y2 = x1, y1
        area = self.area
        x, y = list(self.centroid.coords)[0]
        I_xx = I_x - area * y ** 2
        I_yy = I_y - area * x ** 2
        return I_xx, I_yy


class BeamSection:
    def __init__(self, polygons=None):
        self.polygons = polygons
        self.centroid = None
        self.E = None
        self.G = None
        self.A = None
        self.As = None
        self.Ixx = None
        self.Iyy = None

    @property
    def EIxx(self):
        return self.E * self.Ixx

    @property
    def EA(self):
        return self.E * self.A

    @property
    def GA(self):
        return self.G * self.As

    @property
    def cg(self):
        return self.centroid.coords[0]

    @property
    def xcg(self):
        return self.cg[0]

    @property
    def ycg(self):
        return self.cg[1]

    def analyze(self):
        ESx = ESy = EA = EIxx = EIyy = 0
        A = As = GA = 0

        for p in self.polygons:
            Sx, Sy = p.first_moment

            ESx += p.Ex * Sx
            ESy += p.Ex * Sy

            EA += p.EAx
            GA += p.GA
            A += p.surface
            As += p.As

        ycg = ESx / EA
        xcg = ESy / EA
        E = EA / A

        if As != 0:
            G = GA / As
        else:
            G = 100e9  # ??

        for p in self.polygons:
            A_ = p.surface

            xcg_, ycg_ = p.centroid.coords[0]
            Ix_, Iy_ = p.second_moment
            Ixx_ = Ix_ + A_ * (ycg_ - ycg) ** 2
            Iyy_ = Iy_ + A_ * (xcg_ - xcg) ** 2

            EIxx += p.Ex * Ixx_
            EIyy += p.Ex * Iyy_

        Ixx = EIxx / E
        Iyy = EIyy / E

        self.centroid = geom.Point((xcg, ycg))

        self.E = E
        self.G = G
        self.A = A
        self.As = As
        self.Ixx = Ixx
        self.Iyy = Iyy
