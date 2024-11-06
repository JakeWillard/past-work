
import numpy as np
import sympy as sp
from scipy.sparse import kron, eye, coo_matrix
from numba import njit, jit
import matplotlib.pyplot as plt


class RingGeometry:

    def __init__(self, h, th):

        self.h = h
        self.th = th

    def classify_point(self, x, y, dr, dt):

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, -x)

        if r < 1 - self.h - dr:
            # ignore point
            return False, None
        elif r < 1 - self.h:
            # core ghost point
            return True, "core-ghost"
        elif r < 1 - self.h / 2:
            # interior point
            return True, "interior"
        elif (r < (1 - self.h / 2) + dr) and (np.abs(theta) < self.th - dt):
            # wall ghost point
            return True, "wall-ghost"
        elif r < 1:
            # might be interior or plate ghost point
            if (theta > 0) and (theta < self.th - dt):
                # ignore
                return False, None
            elif (theta > 0) and (theta < self.th):
                # positive plate ghost
                return True, "plate-ghost-pos-flux"
            elif (theta < 0) and (-theta < self.th - dt):
                # ignore
                return False, None
            elif (theta < 0) and (-theta < self.th):
                # negative plate ghost
                return True, "plate-ghost-neg-flux"
            else:
                # interior point
                return True, "interior"
        elif r < 1 + dr:
            # wall ghost
            return True, "wall-ghost"
        else:
            # ignore
            return False, None

    def penalization_vector(self, x, y, dr, dt):

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, -x)

        h_core = 0.5 * (1 + np.math.tanh((1 - self.h - r) / dr))
        if np.abs(theta) < self.th:
            h_wall = 0.5 * (1 + np.math.tanh((r - (1 - self.h / 2)) / dr))
        else:
            h_wall = 0.5 * (1 + np.math.tanh((r - 1) / dr))
        if r > 1 - self.h / 2:
            h_pos = 0.5 * (1 + np.math.tanh((self.th - theta) / dt))
            h_neg = 0.5 * (1 + np.math.tanh((theta + self.th) / dt))
        else:
            h_pos = 0.0
            h_neg = 0.0

        return np.array([h_core, h_wall, h_pos, h_neg])

    def wall_reflection(self, x, y):

        r = np.sqrt(x**2 + y**2)
        r_ref = 1 - np.abs(r - 1)
        rhat = np.array([x, y]) / r

        return r_ref * rhat

    def core_reflection(self, x, y):

        r = np.sqrt(x**2 + y**2)
        r_ref = (1 - self.h) + np.abs(1 - self.h - r)
        rhat = np.array([x, y]) / r

        return r_ref * rhat

    def pos_reflection(self, x, y):

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, -x)
        theta_ref = self.th + np.abs(theta - self.th)

        return r * np.array([np.cos(theta_ref), np.sin(theta_ref)])

    def neg_reflection(self, x, y):

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, -x)
        theta_ref = -self.th - np.abs(theta + self.th)

        return r * np.array([np.cos(theta_ref), np.sin(theta_ref)])

    def trace_forward(self, x, y, dz):

        # for now, make safety factor 2
        q = 2
        dtheta = dz / q

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) + dtheta

        return r * np.array([np.cos(theta), np.sin(theta)])

    def trace_backward(self, x, y, dz):

        # for now, make safety factor 2
        q = 2
        dtheta = dz / q

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) - dtheta

        return r * np.array([np.cos(theta), np.sin(theta)])


class Stencils:

    def __init__(self, mx, my):

        # compute integer off-set for center points of splines
        xc = int(np.ceil(mx / 2))
        yc = int(np.ceil(my / 2))

        # compute and invert vandermonde matrix for x
        Vx = np.empty((mx, mx))
        for i in range(0, mx):
            for j in range(0, mx):
                Vx[i, j] = (i - xc)**j / np.math.factorial(j)
        self._Vxinv = np.linalg.inv(Vx)

        # compute and invert vandermonde matrix for y
        Vy = np.empty((my, my))
        for i in range(0, my):
            for j in range(0, my):
                Vy[i, j] = (i - yc)**j / np.math.factorial(j)
        self._Vyinv = np.linalg.inv(Vy)

        # compute and invert 2d vandermonde matrix
        Vxy = np.empty((mx * my, mx * my))
        for i in range(0, mx):
            for j in range(0, my):
                row = i + j * mx
                for ii in range(0, mx):
                    for jj in range(0, my):
                        col = ii + jj * mx
                        a = (i - xc)**ii / np.math.factorial(ii)
                        b = (j - yc)**jj / np.math.factorial(jj)
                        Vxy[row, col] = a * b
        self._VxyinvT = np.linalg.inv(Vxy).T

        # set other attributes for internal use
        self._mx = mx
        self._my = my
        self._xc = xc
        self._yc = yc

    def interpolation_row(self, x, y, x0, y0, dx, dy, nx, ny):

        # bin onto grid
        i = int((x - x0) // dx)
        j = int((y - y0) // dy)

        # compute relative position in units of dx and dy
        xr = (x - x0) / dx - i
        yr = (y - y0) / dy - j

        # initialize data and coordinate arrays
        dat = np.empty(self._mx * self._my)
        i_vec = np.zeros(self._mx * self._my, dtype=np.int32)
        j_vec = np.zeros(self._mx * self._my, dtype=np.int32)

        # compute dat
        v = np.empty(self._mx * self._my)
        for ii in range(0, self._mx):
            for jj in range(0, self._my):
                k = ii + self._mx * jj
                a = xr**ii / np.math.factorial(ii)
                b = yr**jj / np.math.factorial(jj)
                dat[k] = a * b
        dat = np.dot(self._VxyinvT, v)

        # compute j_vec
        i_bl = i - self._xc
        j_bl = j - self._yc
        for jj in range(0, self._my):
            k1 = i_bl + nx * (j_bl + jj)
            k2 = k1 + self._mx
            j_vec[jj * self._mx:(jj + 1) *
                  self._mx] = np.arange(k1, k2, 1, dtype=np.int32)

        # make row as sparse matrix
        row = coo_matrix((dat, (i_vec, j_vec)), shape=(1, nx * ny))

        return row

    def x_derivative_matrix(self, m, nx, ny):

        # get stencil
        s = self._Vxinv[m, :]

        # get mx and xc
        mx = self._mx
        xc = self._xc

        # set each element by adding together diagonal matrices
        D_1d = np.zeros((nx, nx))
        for i in range(0, mx):
            k = i - xc
            D_1d
            diag_elements = s[i] * np.ones(nx - abs(k))
            D_1d = D_1d + np.diag(diag_elements, k)

        # use kronecker products to compute full 2D matrix
        return kron(eye(ny), D_1d)

    def y_derivative_matrix(self, m, nx, ny):

        # get stencil
        s = self._Vyinv[m, :]

        # get my and yc
        my = self._my
        yc = self._yc

        # set each element by adding together diagonal matrices
        D_1d = np.zeros((ny, ny))
        for i in range(0, my):
            k = i - yc
            D_1d
            diag_elements = s[i] * np.ones(ny - abs(k))
            D_1d = D_1d + np.diag(diag_elements, k)

        # use kronecker products to compute full 2D matrix
        return kron(D_1d, eye(nx))


class RecGrid(Stencils):

    def __init__(self, xs, ys, mx, my):

        # inheret from Stencils class
        super().__init__(mx, my)

        # set xs and ys to attributes
        self.xs = xs
        self.ys = ys

        # set attributes for private use only
        self._nx = len(xs)
        self._ny = len(ys)
        self._x0 = xs[0]
        self._y0 = ys[0]
        self._dx = xs[1] - xs[0]
        self._dy = ys[1] - ys[0]

    def interpolation_row(self, x, y):

        row = super().interpolation_row(x, y, self._x0, self._y0,
                                        self._dx, self._dy, self._nx, self._ny)
        return row

    def x_derivative_matrix(self, m):

        return super().x_derivative_matrix(m, self._nx, self._ny)

    def y_derivative_matrix(self, m):

        return super().y_derivative_matrix(m, self._nx, self._ny)


class RingGrid(RecGrid):

    def __init__(self, geometry, xs, ys, mx, my, dr, dt):

        # inheret from RecGrid
        super().__init__(xs, ys, mx, my)

        # initialize projection matrix from super-grid to ring-grid
        # Proj = np.zeros((self._nx * self._ny, self._nx * self._ny))

        # initialize coordinates for projection matrix
        proj_i = []
        proj_j = []

        # initialize dictionary to organize the relevant points by their classification
        points = {"all": [], "interior": [], "core-ghost": [], "wall-ghost": [],
                  "plate-ghost-pos-flux": [], "plate-ghost-neg-flux": []}

        # classify all the points and assemble projection
        k = 0
        for i in range(0, self._nx):
            for j in range(0, self._ny):

                k_super = i + self._nx * j
                keep, key = geometry.classify_point(xs[i], ys[j], dr, dt)
                if keep:
                    points["all"].append((k, xs[i], ys[j]))
                    points[key].append((k, xs[i], ys[j]))
                    proj_i.append(i)
                    proj_j.append(j)
                    # Proj[k, k_super] = 1
                    k += 1

        # compute sparse projection matrices as private attributes
        self._Pes = coo_matrix(
            (np.ones(k), (proj_i, proj_j)), shape=(k, self._nx * self._ny))
        # self._Pes = coo_matrix(Proj[0:k, :])
        self._Pse = self._Pes.T

        # assign new attributes
        self.geometry = geometry
        self.points = points
        self.N = k

    def interpolation_row(self, x, y):

        # compute rectangular version
        row = super().interpolation_row(x, y)

        # project onto edge-grid
        return coo_matrix(self._Pes.dot(row.T))

    def x_derivative_matrix(self, m):

        # compute rectangular version
        D = super().x_derivative_matrix(m)

        # project onto edge-grid
        return coo_matrix(self._Pes.dot(D.dot(self._Pse)))

    def y_derivative_matrix(self, m):

        # compute rectangular version
        D = super().y_derivative_matrix(m)

        # project onto edge-grid
        return coo_matrix(self._Pes.dot(D.dot(self._Pse)))


class GDBGrid(RingGrid):

    def REF(self):

        # compute core reflection
        dat = []
        i = []
        j = []
        for k, x, y in self.points["core-ghost"]:
            xp, yp = self.geometry.core_reflection(x, y)
            row = self.interpolation_row(xp, yp)
            data = row.data
            r = k * np.ones(len(data), dtype=np.int32)
            c = row.col
            dat.extend(data)
            i.extend(r)
            j.extend(c)
        REF_core = coo_matrix((dat, (i, j)), shape=(self.N, self.N))
        nnz_core = len(dat)

        # compute wall reflection
        dat = []
        i = []
        j = []
        for k, x, y in self.points["wall-ghost"]:
            xp, yp = self.geometry.wall_reflection(x, y)
            row = self.interpolation_row(xp, yp)
            data = row.data
            r = k * np.ones(len(data), dtype=np.int32)
            c = row.col
            dat.extend(data)
            i.extend(r)
            j.extend(c)
        REF_wall = coo_matrix((dat, (i, j)), shape=(self.N, self.N))
        nnz_wall = len(dat)

        # compute positive flux reflection
        dat = []
        i = []
        j = []
        for k, x, y in self.points["plate-ghost-pos-flux"]:
            xp, yp = self.geometry.pos_reflection(x, y)
            row = self.interpolation_row(xp, yp)
            data = row.data
            r = k * np.ones(len(data), dtype=np.int32)
            c = row.col
            dat.extend(data)
            i.extend(r)
            j.extend(c)
        REF_pos = coo_matrix((dat, (i, j)), shape=(self.N, self.N))
        nnz_pos = len(dat)

        # compute negative flux reflection
        dat = []
        i = []
        j = []
        for k, x, y in self.points["plate-ghost-neg-flux"]:
            xp, yp = self.geometry.neg_reflection(x, y)
            row = self.interpolation_row(xp, yp)
            data = row.data
            r = k * np.ones(len(data), dtype=np.int32)
            c = row.col
            dat.extend(data)
            i.extend(r)
            j.extend(c)
        REF_neg = coo_matrix((dat, (i, j)), shape=(self.N, self.N))
        nnz_neg = len(dat)

        # compute nnz_REF vector
        nnz_REF = np.array([nnz_core, nnz_wall, nnz_pos, nnz_neg])

        # compute REF matrix
        M = np.zeros((3, max(nnz_REF), 4))
        M[0, 0:nnz_core, 0] = REF_core.data
        M[1, 0:nnz_core, 0] = REF_core.row
        M[2, 0:nnz_core, 0] = REF_core.col
        M[0, 0:nnz_wall, 1] = REF_wall.data
        M[1, 0:nnz_wall, 1] = REF_wall.row
        M[2, 0:nnz_wall, 1] = REF_wall.col
        M[0, 0:nnz_pos, 2] = REF_pos.data
        M[1, 0:nnz_pos, 2] = REF_pos.row
        M[2, 0:nnz_pos, 2] = REF_pos.col
        M[0, 0:nnz_neg, 3] = REF_neg.data
        M[1, 0:nnz_neg, 3] = REF_neg.row
        M[2, 0:nnz_neg, 3] = REF_neg.col

        return M, nnz_REF

    def TR1(self, dz):

        dat = []
        i = []
        j = []
        for k, x, y in self.points["all"]:
            xp, yp = self.geometry.trace_forward(x, y, dz)
            row = self.interpolation_row(xp, yp)
            data = row.data
            r = k * np.ones(len(data), dtype=np.int32)
            c = row.col
            dat.extend(data)
            i.extend(r)
            j.extend(c)

        M = np.zeros((3, len(dat)))
        M[0, :] = dat
        M[1, :] = i
        M[2, :] = j

        return M

    def TR2(self, dz):

        dat = []
        i = []
        j = []
        for k, x, y in self.points["all"]:
            xp, yp = self.geometry.trace_backward(x, y, dz)
            row = self.interpolation_row(xp, yp)
            data = row.data
            r = k * np.ones(len(data), dtype=np.int32)
            c = row.col
            dat.extend(data)
            i.extend(r)
            j.extend(c)

        M = np.zeros((3, len(dat)))
        M[0, :] = dat
        M[1, :] = i
        M[2, :] = j

        return M

    def PEN(self, dr, dt):

        M = np.zeros((self.N, 4))

        for k, x, y in self.points["all"]:
            M[k, :] = self.geometry.penalization_vector(x, y, dr, dt)

        return M

    def partial_x(self):

        D = self.x_derivative_matrix(1)
        M = np.zeros((3, len(D.data)))
        M[0, :] = D.data
        M[1, :] = D.row
        M[2, :] = D.col
        return M

    def partial_xx(self):

        D = self.x_derivative_matrix(2)
        M = np.zeros((3, len(D.data)))
        M[0, :] = D.data
        M[1, :] = D.row
        M[2, :] = D.col
        return M

    def partial_xy(self):

        Dx = self.x_derivative_matrix(1)
        Dy = self.y_derivative_matrix(1)
        Dxy = coo_matrix(Dx.dot(Dy))
        M = np.zeros((3, len(Dxy.data)))
        M[0, :] = Dxy.data
        M[1, :] = Dxy.row
        M[2, :] = Dxy.col
        return M

    def partial_y(self):

        D = self.y_derivative_matrix(1)
        M = np.zeros((3, len(D.data)))
        M[0, :] = D.data
        M[1, :] = D.row
        M[2, :] = D.col
        return M

    def partial_yy(self):

        D = self.y_derivative_matrix(2)
        M = np.zeros((3, len(D.data)))
        M[0, :] = D.data
        M[1, :] = D.row
        M[2, :] = D.col
        return M
