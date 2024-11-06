

import numpy as np
import sympy as sp
from numba import njit, jit
from scipy.optimize import fmin
from scipy.sparse import kron, eye, coo_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py


# SECTION 1
# PURE GEOMETRY
# =============


class DiverterTransforms:

    """
    Linear transformations associated with diverter plates.

    Instance Attributes:
        m (float): slope of diverter plates in the new coordiantes
        x0 (float): x coordinate for intersection point
        y0 (float): y coordiante for intersection point
    """

    def __init__(self, L, theta, m, ep, d, ka, r):
        """
        Constructor for DiverterTransforms class.

        Args:
            L (float): distance from x-point to the intersection point
            theta (float): angle w.r.t y-axis of line between x-point and intersection point
            m (float): value for instance attribute m
            ep (float): aspect ratio
            d (float): triangulation
            ka (float): elongation
            r (float): shift parameter that locates the x-points
        """

        # compute location of x-point
        xn = 1 - d * ep * (1 + r)
        yn = ka * ep * (1 + r)

        # compute dispacement vector between x-point and where the
        # plates intersect
        dr = L * np.array([-np.sin(theta), np.cos(theta)])

        # set attributes
        self.m = m
        [self.x0, self.y0] = [xn, yn] + dr

        # compute matrix that rotates clockwise by angle theta
        self._Rot = np.array([[np.cos(theta), np.sin(theta)],
                              [-np.sin(theta), np.cos(theta)]])

        # for a line passing through the origin with slope +/-m, compute
        # matrix that reflects points across the line for both cases +/-
        self._R_pos = np.array([[1 - m**2, 2 * m],
                                [2 * m, m**2 - 1]]) / (1 + m**2)

        self._R_neg = np.array([[1 - m**2, -2 * m],
                                [-2 * m, m**2 - 1]]) / (1 + m**2)

    def Q(self, x, y, y_sign):
        """
        Coordinate transform into diverter-relative coordinates.

        In these coordinates, the origin is placed at the point of intersection (x0, y0), and
        the x-point exists at (0, -L). The diverter plates are then two intersecting lines through
        the origin y = +/- m*x. There is also a reflection about the y-axis in the lower diverter case,
        so that the magnetic flux is positive into the right plate (y = -m*x) and is negative
        into the left plate (y = +m*x).

        Args:
            x (float): x coordinate in original coordinates
            y (float): y coordinate in original coordinates
            y_sign (int): +1 if upper diverter, -1 if lower diverter

        Returns:
            (float, float): the point (x, y) in diverter-relative coordinate system
        """

        # shift (x0, y0) to the origin and reflect about x-axis
        # also has the effect of a parity transformation if y_sign = -1
        xr = y_sign * (x - self.x0)
        yr = y_sign * y - self.y0

        # return clockwise rotation
        return np.dot(self._Rot, [xr, yr])

    def Qinv(self, xr, yr, y_sign):
        """
        Exactly the inverse operation as the transformation Q.

        Args:
            xr (float): x coordinate in diverter-relative coordinates
            yr (float): y coordinate in diverter-relative coordinates
            y_sign (int): +1 if upper diverter, -1 if lower diverter

        Returns:
            (float, float): the point (xr, yr) in original coordinate system
        """

        # apply counter-clockwise rotation and apply parity transform if y_sign = -1
        r = y_sign * np.dot(self._Rot.T, np.array([xr, yr]))

        # return shifting (x0, y0) back to original position
        return r + [self.x0, self.y0]

    def reflect(self, x, y, y_sign):
        """
        Reflection about diverter plate.

        Transforms into diverter-relative coordinates, then uses the formula for
        reflection about a line intersecting the origin. The reflection is carried out
        w.r.t whichever plate is relevant given the sign of x in this coordinate system:
        reflect about y = -m*x if x > 0, and about y = m*x if x < 0.

        Args:
            x (float): x coordinate in original coordinates
            y (float): y coordinate in original coordinates
            y_sign (int): +1 if upper diverter, -1 if lower diverter

        Returns:
            (float, float): the transformed point in the original coordinates
        """

        # apply Q-transformation
        r = self.Q(x, y, y_sign)

        # reflect differently depending on the sign of xr
        if r[0] > 0:
            xr, yr = np.dot(self._R_neg, r)
        else:
            xr, yr = np.dot(self._R_pos, r)

        # return in original coordinates
        return self.Qinv(xr, yr, y_sign)

    def F(self, x, y, flux_sign, y_sign):
        """
        Distance function w.r.t chosen diverter plate.

        In diverter-relative coordinates, the positive-flux plate is
        a line through the origin with slope -m, while the negative-flux plate
        is a line with slope +m. This function computes y +/- m*x in those coordinates
        to produce a value that is zero on the chosen plate.

        Args:
            x (float): x coordinate in original coordinates
            y (float): y coordinate in original coordinates
            flux_sign (int): +1 for positive-flux plate, -1 for negative-flux plate
            y_sign (int): +1 if upper diverter, -1 if lower diverter

        Returns:
            float: value of y +/- m*x
        """

        # apply Q-transformation
        xr, yr = self.Q(x, y, y_sign)

        # return distance above the line
        return yr + (flux_sign * self.m) * xr


def cerfon_flux(x, y, A, C):
    """
    Defines general solution to Grad-Shafranov from Cerfon et al 2010.

    Defines a function of x and y with parameters A and C according to the
    expansion detailed in Cerfon's paper which solves the Grad-Shafranov equation
    with Solov'ev profiles.

    Args:
        x (symbol): x coordinate
        y (symbol): y coordinate
        A (symbol): constant that determines the B_phi profile
        C (indexed symbol): coefficients for expansion

    Returns:
        expression: general solution for flux function
    """

    # define vector for homogenous basis function
    psi_H = sp.zeros(1, 12)

    # define particular and homogenous terms
    psi_p = (1 - A) * x**4 / 8 + A * x**2 * sp.log(x) / 2
    psi_H[0, 0] = 1
    psi_H[0, 1] = x**2
    psi_H[0, 2] = y**2 - x**2 * sp.log(x)
    psi_H[0, 3] = x**4 - 4 * x**2 * y**2
    psi_H[0, 4] = 2 * y**4 - 9 * y**2 * x**2 + 3 * x**4 * \
        sp.log(x) - 12 * x**2 * y**2 * sp.log(x)
    psi_H[0, 5] = x**6 - 12 * x**4 * y**2 + 8 * x**2 * y**4
    psi_H[0, 6] = 8 * y**6 - 140 * y**4 * x**2 + 75 * y**2 * x**4 - 15 * x**6 * \
        sp.log(x) + 180 * x**4 * y**2 * sp.log(x) - \
        120 * x**2 * y**4 * sp.log(x)
    psi_H[0, 7] = y
    psi_H[0, 8] = y * x**2
    psi_H[0, 9] = y**3 - 3 * y * x**2 * sp.log(x)
    psi_H[0, 10] = 3 * y * x**4 - 4 * y**3 * x**2
    psi_H[0, 11] = 8 * y**5 - 45 * y * x**4 - 80 * y**3 * \
        x**2 * sp.log(x) + 60 * y * x**4 * sp.log(x)

    # return general solution
    return psi_p + np.sum([C[i] * psi_H[0, i] for i in range(0, 12)])


def fit_geometry(x, y, psi, C, ep, de, ka, r, x_up, x_down):
    """
    Places geometric constraints on psi and solves for coefficients C.

    Fixes value and derivatives of psi at specific points to determine
    coefficients C that produce a desired flux function. This essentially is
    doing the same calculations discussed in Cerfon et al 2010. Returns a list of
    sympy substutution tuples for expressions that depend on C.

    Args:
        x (symbol): x coordinate
        y (symbol): y coordinate
        psi (expression): general solution for flux function, depends on (x, y, C)
        C (indexed symbol): coefficients for general solution
        ep (float): aspect ratio
        de (float): triangulation
        ka (float): elongation
        r (float): shift parameter that locates the x-points
        x_up (boolean): True for upper x-point, False for no upper x-point
        x_down (boolean): True for lower x-point, False for no lower x-point


    Returns:
        list: tuples (C[i], C_solved[i]) for each i
    """

    # compute x-point shifts
    dx = de * ep * (1 + r)
    dy = ka * ep * (1 + r)

    # compute subs lists for points of interest
    a = [(x, 1 + ep), (y, 0)]
    b = [(x, 1 - de * ep), (y, ka * ep)]
    c = [(x, 1 - ep), (y, 0)]
    d = [(x, 1 - de * ep), (y, -ka * ep)]
    xp1 = [(x, 1 - dx), (y, dy)]
    xp2 = [(x, 1 - dx), (y, -dy)]

    # compute N values
    alpha = np.arcsin(de)
    N1 = -(1 + alpha)**2 / (ep * ka**2)
    N2 = (1 - alpha)**2 / (ep * ka**2)
    N3 = -ka / (ep * np.cos(alpha)**2)

    # initialize list of constraints
    eqns = []

    # add inboard and outboard side constraints
    eqns.append(psi.subs(a))
    eqns.append(psi.diff(y, 1).subs(a))
    eqns.append((psi.diff(y, 2) + N1 * psi.diff(x, 1)).subs(a))
    eqns.append(psi.subs(c))
    eqns.append(psi.diff(y, 1).subs(c))
    eqns.append((psi.diff(y, 2) + N2 * psi.diff(x, 1)).subs(c))

    # If upper x-point, set B=0 at x-point. Otherwise,
    # set direction and curvature of core-side surface
    if x_up:
        eqns.append(psi.subs(xp1))
        eqns.append(psi.diff(x, 1).subs(xp1))
        eqns.append(psi.diff(y, 1).subs(xp1))
    else:
        eqns.append(psi.subs(b))
        eqns.append(psi.diff(x, 1).subs(b))
        eqns.append((psi.diff(x, 2) + N3 * psi.diff(y, 1)).subs(b))

    # If lower x-point, set B=0 at x-point. Otherwise,
    # set direction and curvature of core-side surface
    if x_down:
        eqns.append(psi.subs(xp2))
        eqns.append(psi.diff(x, 1).subs(xp2))
        eqns.append(psi.diff(y, 1).subs(xp2))
    else:
        eqns.append(psi.subs(d))
        eqns.append(psi.diff(x, 1).subs(d))
        eqns.append((psi.diff(x, 2) - N3 * psi.diff(y, 1)).subs(d))

    # solve for C
    Cs = list(sp.linsolve(eqns, [C[i] for i in range(0, 12)]))[0]

    # return list of substitutions for C
    return [(C[i], Cs[i]) for i in range(0, 12)]


class EdgeGeometry:

    """
    Class for specifying the shape of the edge region.

    Using the solution to Grad-Shafranov described in Cerfon et al 2010, boundaries
    for the edge region are specified in terms of flux surfaces. A reduced radial coordinate
    is calculated which is 0 on the core-side surface and 1 on the wall and dome surfaces.
    This in combination with DiverterTransforms instances allows for the classification of specific
    points as either exterior points (irrelevant for edge simulations), ghost points (outside the boundaries
    but needed to set boundary conditions), or interior points (where relevant quantities will be simulated).

    Class Attributes:
        x (symbol): x coordinate
        y (symbol): y coordinate
        A (symbol): constant that determines the B_phi profile
        B0 (symbol): constant that determines the B_phi profile
        C (indexed symbol): coefficients for general solution
        psi_G (expression): general solution for flux function
        Bx (expression): x component of magnetic field
        Bx (expression): y component of magnetic field
        Bx (expression): z component of magnetic field
        B (expression): magnetic field strength
        bx (expression): x component of magnetic field unit vector
        by (expression): y component of magnetic field unit vector
        bz (expression): z component of magnetic field unit vector
        q (expression): kink safety factor

    Instance Attributes:
        epsilon (float): aspect ratio
        delta (float): triangulation
        kappa (float): elongation
        yX (float): |y| at the separatrix
        h (float): thickness of edge on in-board side
        pC (float): value of psi on core-side boundary
        pW (float): value of psi on wall boundary
        pD (float): value of psi on dome boundary
        pX (float): value of psi on seperatrix
        upper_diverter_active (boolean): True if there is an upper x-point/diverter
        lower_diverter_active (boolean): True if there is a lower x-point/diverter
        diverter_transforms (DiverterTransforms): diverter transforms for selected geometry
        psi_jit (function): numba compiled flux function
        bx_jit (function): numba compiled x component of magnetic field unit vector
        by_jit (function): numba compiled y component of magnetic field unit vector
    """

    # define magnetic field parameters
    x, y = sp.symbols("x y")
    C = sp.IndexedBase("C")
    A, B0 = sp.symbols("A B0")

    # define general solution and components of B
    psi_G = cerfon_flux(x, y, A, C)
    Bx = -psi_G.diff(y, 1) / x
    By = psi_G.diff(x, 1) / x
    Bz = sp.sqrt(B0**2 - 2 * A * psi_G) / x

    # define safety factor
    q = sp.sqrt(x**2 + y**2) * Bz / (Bx**2 + By**2)

    # define field strength and unit vectors
    B = sp.sqrt(Bx**2 + By**2 + Bz**2)
    bx = Bx / B
    by = By / B
    bz = Bz / B

    def __init__(self, A, B0, h, psi_params, div_params, x_up, x_down):
        """
        Constructor for EdgeGeometry class.

        Args:
            A (float): constant that determines the B_phi profile
            B0 (float): constant that determines the B_phi profile
            h (float): sets instance attribute h
            psi_params (array): args for fit_geometry function
            div_params (array): args for DiverterTransforms instances.
            x_up (boolean): True for upper x-point, False for no upper x-point
            x_down (boolean): True for lower x-point, False for no lower x-point
        """

        # unpack parameters
        ep, d, ka, r = psi_params
        L, theta, m = div_params

        # make list of substitutions for psi
        psi_subs = fit_geometry(self.x, self.y, self.psi_G,
                                self.C, ep, d, ka, r, x_up, x_down)
        psi_subs.append((self.A, A))
        psi_subs.append((self.B0, B0))

        # make substitutions
        psi = self.psi_G.subs(psi_subs)
        bx = self.bx.subs(psi_subs)
        by = self.by.subs(psi_subs)
        bz = self.bz.subs(psi_subs)

        # lambdify psi now for intermediate calculations
        psi_lam = sp.lambdify((self.x, self.y), psi)

        # if an x-point was added in, then psi=0 defines the seperatrix.
        # otherwise, it defines the core-side boundary.
        if x_up or x_down:
            pX = 0
            pC = psi_lam(1 - ep + h / 2, 0)
            pW = psi_lam(1 - ep - h / 2, 0)
        else:
            pX = psi_lam(1 - ep - h / 2, 0)
            pC = 0
            pW = psi_lam(1 - ep - h, 0)

        # initialize diverter transforms
        div = DiverterTransforms(L, theta, m, ep, d, ka, r)

        # compute psi for the dome flux surface (calculation isn't meaningful
        # and isn't used if there is no x-point)
        [xd1, yd1] = div.Qinv(0, -L / 2, 1)
        [xd2, yd2] = div.Qinv(0, -L / 2, -1)
        pD = np.min([psi_lam(xd1, yd1), psi_lam(xd2, yd2)])

        # set attributes
        self.epsilon = ep
        self.delta = d
        self.kappa = ka
        self.yX = ep * ka * (1 + r)
        self.h = h
        self.pC = pC
        self.pW = pW
        self.pD = pD
        self.pX = pX
        self.upper_diverter_active = x_up
        self.lower_diverter_active = x_down
        self.diverter_transforms = div

        # jit compiled callable attributes
        self.psi_jit = njit(psi_lam)
        self.bx_jit = njit(sp.lambdify((self.x, self.y), bx))
        self.by_jit = njit(sp.lambdify((self.x, self.y), by))
        self.bz_jit = njit(sp.lambdify((self.x, self.y), bz))

    def classify_point(self, x, y, dr, dp):
        """
        Gives a label to the point (x, y) depending on its relationship to the geometry.

        Points can be classified as (1) core-side ghost, (2) interior points, (3) wall-side ghost,
        (4) diverter plate ghost with positive flux, (5) diverter plate ghost with negative flux, and
        (6) diverter dome ghost.

        Args:
            x (float): x coordinate
            y (float): y coordinate
            dr (float): min variation in distance above diverter plate for a point to be ignored
            dp (float): min variation in flux past a flux boundary for a point to be ignored

        Returns:
            (boolean, string): (whether or not to keep the point, classification flag)
        """

        # 5 values needed to oriente (x, y) w.r.t the various surfaces:
        # the value of psi, and the distance functions for positive and negative
        # flux diverter plates in the upper and lower configurations.
        p = self.psi_jit(x, y)
        f_pos_upper = self.diverter_transforms.F(x, y, 1, 1)
        f_neg_upper = self.diverter_transforms.F(x, y, -1, 1)
        f_pos_lower = self.diverter_transforms.F(x, y, 1, -1)
        f_neg_lower = self.diverter_transforms.F(x, y, -1, -1)

        # complicated web of if statements to determine the classification
        if (p < self.pD - dp) and (np.abs(y) > self.yX):
            # ignore point
            return False, None

        elif (p < self.pD) and (np.abs(y) > self.yX):
            # ghost point behind dome
            return True, "dome-ghost"

        elif (p < self.pC - dp) and (np.abs(y) <= self.yX):
            # ignore point
            return False, None

        elif (p < self.pC) and (np.abs(y) <= self.yX):
            # ghost point in core-region
            return True, "core-ghost"

        elif p < self.pW:
            # either an interior point or a ghost point for diverter plates

            # check upper diverter config
            if self.upper_diverter_active:
                if f_pos_upper > dr:
                    # ignore point
                    return False, None

                elif f_pos_upper > 0:
                    # ghost point for positive-flux diverter plate
                    return True, "plate-ghost-pos-flux"

                elif f_neg_upper > dr:
                    # ignore point
                    return False, None

                elif f_neg_upper > 0:
                    # ghost point for negative flux diverter plate
                    return True, "plate-ghost-neg-flux"

                else:
                    # interior point
                    return True, "interior"

            # check lower diverter config
            elif self.lower_diverter_active:
                if f_pos_lower > dr:
                    # ignore point
                    return False, None

                elif f_pos_lower > 0:
                    # ghost point for positive flux ghost point
                    return True, "plate-ghost-pos-flux"

                elif f_neg_lower > dr:
                    # ignore point
                    return False, None

                elif f_neg_lower > 0:
                    # ghost point for negative flux diverter plate
                    return True, "plate-ghost-neg-flux"

                else:
                    return True, "interior"

            else:
                # if no diverter configuration, this is an interior point
                return True, "interior"

        elif p < self.pW + dp:
            # wall ghost point
            return True, "wall-ghost"

        else:
            # ignore point
            return False, None


# SECTION 2
# RECTANGULAR FINITE DIFFERENCE
# =============================


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

        # get mx, my, xc, and yc
        mx = self._mx
        my = self._my
        xc = self._xc
        yc = self._yc

        # initialize vector and matrix for stencil
        s_vec = np.empty(mx * my)
        s_mat = np.empty((mx, my))

        # compute s_vec
        for ii in range(0, mx):
            for jj in range(0, my):
                k = ii + mx * jj
                a = xr**ii / np.math.factorial(ii)
                b = yr**jj / np.math.factorial(jj)
                s_vec[k] = a * b

        # reshape to make s_mat
        s_mat = s_vec.reshape((my, mx))

        # compute full-sized row
        row = np.zeros(nx * ny)
        for jj in range(my):
            k1 = (i - xc) + nx * (j - yc + jj)
            k2 = k1 + mx
            row[k1:k2] = s_mat[jj, :]

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

    def interpolation_matrix(self, points):

        N = size(points)[0]
        mat = np.empty((N, self._nx * self.ny))

        # assemble mat row by row
        for k in range(0, N):
            x, y = points[k, :]
            mat[k, :] = self.interpolation_row(x, y)

        # return sparse matrix
        return coo_matrix(mat)


# SECTION 3
# FINITE DIFFERENCE FOR EDGE REGION
# =================================

class EdgeGrid(RecGrid):

    def __init__(self, geometry, xs, ys, mx, my, dr, dp):

        # inheret from RecGrid
        super().__init__(xs, ys, mx, my)

        # initialize projection matrix from super-grid to edge-grid
        Proj = np.zeros((self._nx * self._ny, self._nx * self._ny))

        # initialize dictionary to organize the relevant points by their classification
        points = {"all": [], "interior": [], "core-ghost": [], "wall-ghost": [],
                  "plate-ghost-pos-flux": [], "plate-ghost-neg-flux": [],
                  "dome-ghost": []}

        # classify all the points and assemble projection
        k = 0
        for i in range(0, self._nx):
            for j in range(0, self._ny):

                k_super = i + self._nx * j
                keep, key = geometry.classify_point(xs[i], ys[j], dr, dp)
                if keep:
                    points["all"].append((k, xs[i], ys[j]))
                    points[key].append((k, xs[i], ys[j]))
                    Proj[k, k_super] = 1
                    k += 1

        # compute sparse projection matrices as private attributes
        self._Pes = coo_matrix(Proj[0:k, :])
        self._Pse = self._Pes.T

        # assign new attributes
        self.geometry = geometry
        self.points = points
        self.N = k

    def interpolation_row(self, x, y):

        # compute rectangular version
        row = super().interpolation_row(x, y)

        # project onto edge-grid
        return self._Pes.dot(row)

    def x_derivative_matrix(self, m):

        # compute rectangular version
        D = super().x_derivative_matrix(m)

        # project onto edge-grid
        return self._Pes.dot(D.dot(self._Pse))

    def y_derivative_matrix(self, m):

        # compute rectangular version
        D = super().y_derivative_matrix(m)

        # project onto edge-grid
        return self._Pes.dot(D.dot(self._Pse))

    def interpolation_matrix(self, points):

        # compute rectangular version
        D = super().interpolation_matrix(points)

        # project onto edge-grid
        return D.dot(self._Pse)


# SECTION 4
# OPERATORS FOR GDB
# =================


@njit
def trace_fieldline(k, x0, y0, bx, by, bz, ds, dz, dir=1):
    """
    Trace fieldline to adjacent z-plane.

    Uses RK4 algorithm to trace fieldline forward until the change in z exceeds dz.

    Args:
        k (int): placement of (x0, y0) in some list of points
        x0 (float): initial x coordinate
        y0 (float): initial y coordinate
        bx (function): x component of B-field unit vector
        by (function): y component of B-field unit vector
        bz (function): z component of B-field unit vector
        ds (float): step size for RK4 integrator
        dz (float): change in z before integration terminates
        dir (int): flag +1/-1 determines parallel tracing or anti-parallel tracing. Default: +1

    Returns:
        (int, float, float, float): (k, final x, final y, total distance traced)
    """

    z = 0
    x = x0
    y = y0
    dL = 0

    while z < dz:

        # perform RK4 iteration
        kx1 = bx(x, y)
        ky1 = by(x, y)
        kz1 = bz(x, y)

        kx2 = bx(x + ds * kx1 / 2, y + ds * ky1 / 2)
        ky2 = by(x + ds * kx1 / 2, y + ds * ky1 / 2)
        kz2 = bz(x + ds * kx1 / 2, y + ds * ky1 / 2)

        kx3 = bx(x + ds * kx2 / 2, y + ds * ky2 / 2)
        ky3 = by(x + ds * kx2 / 2, y + ds * ky2 / 2)
        kz3 = bz(x + ds * kx2 / 2, y + ds * ky2 / 2)

        kx4 = bx(x + ds * kx3, y + ds * ky3)
        ky4 = by(x + ds * kx3, y + ds * ky3)
        kz4 = bz(x + ds * kx3, y + ds * ky3)

        x += ds * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
        y += ds * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6
        z += ds * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6
        dL += ds

    return (k, x, y, dL)


@njit
def trace_gradient(k, x0, y0, bx, by, psi, pb, ds, dir=1):
    """
    Trace flux gradient into the interior region.

    Uses RK4 algorithm to trace along flux gradient until crossing the boundary, then
    continue tracing for an equal distance. The result is analogous to a reflection about
    the boundary. Since this function will be performed on lists of points, the integer k
    representing the placement in the list is returned with the final position so that the
    ordering is easily preserved.

    Args:
        k (int): placement of (x0, y0) in some list of points
        x0 (float): initial x coordinate
        y0 (float): initial y coordinate
        bx (function): x component of B-field unit vector
        by (function): y component of B-field unit vector
        psi (function): flux function
        pb (float): value of psi on boundary
        ds (float): step size for RK4 integrator
        dir (int): flag +1/-1 determines parallel tracing or anti-parallel tracing. Default: +1

    Returns:
        (int, float, float, float): (k, final x, final y, total distance traced)
    """

    x = x0
    y = y0
    l = 0

    # first trace the curve until intersection with psi=b surface
    while dir * (psi(x, y) - pb) < 0:
        kx1 = dir * by(x, y)
        ky1 = -dir * bx(x, y)

        kx2 = dir * by(x + ds * kx1 / 2, y + ds * ky1 / 2)
        ky2 = -dir * bx(x + ds * kx1 / 2, y + ds * ky1 / 2)

        kx3 = dir * by(x + ds * kx2 / 2, y + ds * ky2 / 2)
        ky3 = -dir * bx(x + ds * kx2 / 2, y + ds * ky2 / 2)

        kx4 = dir * by(x + ds * kx3, y + ds * ky3)
        ky4 = -dir * bx(x + ds * kx3, y + ds * ky3)

        x += ds * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
        y += ds * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6
        l += ds

    # the final distance traced is twice the distanced traced so far
    dL = 2 * l

    # continue tracing along the gradient for a distance equal to current l
    while l > 0:
        kx1 = dir * by(x, y)
        ky1 = -dir * bx(x, y)

        kx2 = dir * by(x + ds * kx1 / 2, y + ds * ky1 / 2)
        ky2 = -dir * bx(x + ds * kx1 / 2, y + ds * ky1 / 2)

        kx3 = dir * by(x + ds * kx2 / 2, y + ds * ky2 / 2)
        ky3 = -dir * bx(x + ds * kx2 / 2, y + ds * ky2 / 2)

        kx4 = dir * by(x + ds * kx3, y + ds * ky3)
        ky4 = -dir * bx(x + ds * kx3, y + ds * ky3)

        x += ds * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
        y += ds * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6
        l -= ds

    return (k, x, y, dL)


def find_separatrix(theta, psi, pX):

    # define function to minimize
    f = njit(lambda r: abs(psi(r * cos(theta), r * sin(theta)) - pX))

    # find the point on the separatrix
    r = fmin(f, 0.01)[0]
    x = r * cos(theta)
    y = r * sin(theta)

    return x, y


class GDBGrid(EdgeGrid):

    def flux_average(self, N_angles):

        # initialize kernal
        kernal = np.zeros(self.N)

        # get flux function and value at separatrix
        psi = self.geometry.psi_jit
        pX = self.geometry.pX

        # initialize executor
        with ProcessPoolExecutor(max_workers=nworkers) as exe:

            # create futures set
            futures = set()

            # loop through angles between 0 and 2*pi
            thetas = np.linspace(0, 2 * np.pi, N_angles + 1)
            thetas = thetas[0, N_angles]
            for t in thetas:

                # find point on the separatrix given theta=t
                futures.add(exe.submit(find_separatrix, t, psi, pX))

            # as the points are obtained, add the interpolation rows to the kernal
            for fut in as_completed(futures):

                x, y = fut.result()
                kernal += self.interpolation_row(x, y)

        # divide by N_angles to get average
        return kernal / N_angles

    def plate_reflections(self):

        # initialize reflection matrices
        pos_ref = np.eye(self.N)
        neg_ref = np.eye(self.N)

        # diverter transforms object
        div = self.geometry.diverter_transforms

        # assemble positive flux reflection
        for k, x, y in self.points["plate-ghost-pos-flux"]:
            xr, yr = div.reflect(x, y, np.sign(y))
            pos_ref_rec[k, :] = self.interpolation_row(xr, yr)
        pos_ref = coo_matrix(pos_ref_rec)

        # assemble negative flux reflection
        for k, x, y in self.points["plate-ghost-neg-flux"]:
            xr, yr = div.reflect(x, y, np.sign(y))
            pos_ref_rec[k, :] = self.interpolation_row(xr, yr)
        neg_ref = coo_matrix(neg_ref_rec)

        return pos_ref, neg_ref

    def core_reflection(self, ds, nworkers=1):

        # initialize reflection matrix
        core_ref = np.eye(self.N)

        # get magnetic field unit vectors and flux function
        bx = self.geometry.bx_jit
        by = self.geometry.by_jit
        psi = self.geometry.psi_jit

        # value of flux function on core boundary
        pC = self.geometry.pC

        # initialize executor
        with ProcessPoolExecutor(max_workers=nworkers) as exe:

            # create futures set
            futures = set()

            # trace gradient from each core ghost point
            for k, x, y in self.points["core-ghost"]:
                f = exe.submit(trace_gradient, k, x, y, bx, by,
                               psi, pC, ds, dir=1)
                futures.add(f)

            # get interpolation rows as the gradient tracing is completed
            for fut in as_completed(futures):
                k, x, y, dL = fut.result()
                core_ref[k, :] = self.interpolation_row(x, y)

        return coo_matrix(core_ref)

    def wall_reflection(self, ds, nworkers=1):

        # initialize reflection matrix
        wall_ref = np.eye(self.N)

        # get magnetic field unit vectors and flux function
        bx = self.geometry.bx_jit
        by = self.geometry.by_jit
        psi = self.geometry.psi_jit

        # value of flux function on wall boundary
        pW = self.geometry.pW

        # initialize executor
        with ProcessPoolExecutor(max_workers=nworkers) as exe:

            # create futures set
            futures = set()

            # trace gradient from each wall ghost point
            for k, x, y in self.points["wall-ghost"]:
                f = exe.submit(trace_gradient, k, x, y, bx, by,
                               psi, pW, ds, dir=-1)
                futures.add(f)

            # get interpolation rows as the gradient tracing is completed
            for fut in as_completed(futures):
                k, x, y, dL = fut.result()
                wall_ref[k, :] = self.interpolation_row(x, y)

        return coo_matrix(wall_ref)

    def dome_reflection(self, ds, nworkers=1):

        # initialize reflection matrix
        dome_ref = np.eye(self.N)

        # get magnetic field unit vectors and flux function
        bx = self.geometry.bx_jit
        by = self.geometry.by_jit
        psi = self.geometry.psi_jit

        # value of flux function on dome boundary
        pD = self.geometry.pC

        # initialize executor
        with ProcessPoolExecutor(max_workers=nworkers) as exe:

            # create futures set
            futures = set()

            # trace gradient from each core ghost point
            for k, x, y in self.points["dome-ghost"]:
                f = exe.submit(trace_gradient, k, x, y, bx, by,
                               psi, pD, ds, dir=1)
                futures.add(f)

            # get interpolation rows as the gradient tracing is completed
            for fut in as_completed(futures):
                k, x, y, dL = fut.result()
                dome_ref[k, :] = self.interpolation_row(x, y)

        return coo_matrix(dome_ref)

    def fieldline_maps(self, dz, ds, nworkers=1):

        # get magnetic field unit vectors
        bx = self.geometry.bx_jit
        by = self.geometry.by_jit
        bz = self.geometry.bz_jit

        # initialize matrices
        TR1 = np.zeros((self.N, self.N))
        TR2 = np.zeros((self.N, self.N))

        # dL vectors
        dL1 = np.zeros(self.N)
        dL2 = np.zeros(self.N)

        # initialize executor
        with ProcessPoolExecutor(max_workers=nworkers) as exe:

            # create futures sets for forward and backward maps
            futures_1 = set()
            futures_2 = set()

            # trace field-line from each point
            for k, x, y in self.points["all"]:
                f1 = exe.submit(trace_fieldline, k, x, y,
                                bx, by, bz, ds, dz, dir=1)
                f2 = exe.submit(trace_fieldline, k, x, y,
                                bx, by, bz, ds, dz, dir=-1)

                futures_1.add(f1)
                futures_2.add(f2)

            # as the field-lines as traced, compute interpolation rows
            for f in as_completed(futures_1):
                k, x, y, L = f.result()
                TR1[k, :] = self.interpolation_row(x, y)
                dL1[k] = L
            for f in as_completed(futures_2):
                k, x, y, L = f.result()
                TR2[k, :] = self.interpolation_row(x, y)
                dL2[k] = L

        TR1 = coo_matrix(TR1)
        TR2 = coo_matrix(TR2)

        return TR1, TR2, dL1, dL2

    def penalization_vectors(self):

        # initialize vectors
        pen = np.zeros((self.N, 5))

        # for now, just do crude step functions
        for k, x, y in self.points["core-ghost"]:
            pen[k, 0] = 1
        for k, x, y in self.points["wall-ghost"]:
            pen[k, 1] = 1
        for k, x, y in self.points["plate-ghost-pos-flux"]:
            pen[k, 2] = 1
        for k, x, y in self.points["plate-ghost-neg-flux"]:
            pen[k, 3] = 1
        for k, x, y in self.points["dome-ghost"]:
            pen[k, 4] = 1

        return pen

    def make_output_file(self, path, dz, ds, nworkers=1):

        # make file object. never allow this to overwrite an existing file!
        f = h5py.File(path, "w-")

        # give the file some basic attributes
        f.attrs.create("nxy", self.N)

        # make group for storing the points on the grid and their types
        pg = f.create_group("grid-points")
        pg["all"] = self.points["all"]
        pg["interior"] = self.points["interior"]
        pg["core-ghost"] = self.points["core-ghost"]
        pg["wall-ghost"] = self.points["wall-ghost"]
        pg["plate-ghost-pos-flux"] = self.points["plate-ghost-pos-flux"]
        pg["plate-ghost-neg-flux"] = self.points["plate-ghost-neg-flux"]
        pg["dome-ghost"] = self.points["dome-ghost"]

        # compute penalization vectors
        pg["PEN"] = self.penalization_vectors()

        # make group for operators
        og = f.create_group("Operators")

        # make derivative matrices
        Dx = self.x_derivative_matrix(1)
        Dxx = self.x_derivative_matrix(2)
        Dy = self.y_derivative_matrix(1)
        Dxy = Dx.dot(Dy)
        Dyy = self.y_derivative_matrix(2)

        # for now, approximate the curvature operator as in grillix
        C = -2 * Dy

        # compute reflection matrices
        REF1 = self.core_reflection(ds, nworkers=nworkers)
        REF2 = self.wall_reflection(ds, nworkers=nworkers)
        REF3, REF4 = self.plate_reflections()
        REF5 = self.dome_reflection(ds, nworkers=nworkers)

        # compute field-line tracing
        TR1, TR2, ds1, ds2 = self.fieldline_maps(dz, ds, nworkers=nworkers)

        # write operators to file
        og["Dx"] = np.empty((3, Dx.nnz))
        og["Dx"].attrs["nnz"] = Dx.nnz
        og["Dx"][0, :] = Dx.data
        og["Dx"][1, :] = Dx.row
        og["Dx"][2, :] = Dx.col

        og["Dxx"] = np.empty((3, Dxx.nnz))
        og["Dxx"].attrs["nnz"] = Dxx.nnz
        og["Dxx"][0, :] = Dxx.data
        og["Dxx"][1, :] = Dxx.row
        og["Dxx"][2, :] = Dxx.col

        og["Dxy"] = np.empty((3, Dxy.nnz))
        og["Dxy"].attrs["nnz"] = Dxy.nnz
        og["Dxy"][0, :] = Dxy.data
        og["Dxy"][1, :] = Dxy.row
        og["Dxy"][2, :] = Dxy.col

        og["Dy"] = np.empty((3, Dy.nnz))
        og["Dy"].attrs["nnz"] = Dy.nnz
        og["Dy"][0, :] = Dy.data
        og["Dy"][1, :] = Dy.row
        og["Dy"][2, :] = Dy.col

        og["Dyy"] = np.empty((3, Dyy.nnz))
        og["Dyy"].attrs["nnz"] = Dyy.nnz
        og["Dyy"][0, :] = Dyy.data
        og["Dyy"][1, :] = Dyy.row
        og["Dyy"][2, :] = Dyy.col

        og["C"] = np.empty((3, C.nnz))
        og["C"].attrs["nnz"] = C.nnz
        og["C"][0, :] = C.data
        og["C"][1, :] = C.row
        og["C"][2, :] = C.col

        nnz_REF = np.array([REF1.nnz, REF2.nnz, REF3.nnz, REF4.nnz, REF5.nnz])
        nnz_REF_max = np.max(nnz_REF)
        og["REF"] = np.empty((3, nnz_REF_max, 5))
        og["REF"].attrs["nnz"] = nnz_REF
        og["REF"][0, :, 0] = REF1.data
        og["REF"][1, :, 0] = REF1.row
        og["REF"][2, :, 0] = REF1.col
        og["REF"][0, :, 1] = REF2.data
        og["REF"][1, :, 1] = REF2.row
        og["REF"][2, :, 1] = REF2.col
        og["REF"][0, :, 2] = REF3.data
        og["REF"][1, :, 2] = REF3.row
        og["REF"][2, :, 2] = REF3.col
        og["REF"][0, :, 3] = REF4.data
        og["REF"][1, :, 3] = REF4.row
        og["REF"][2, :, 3] = REF4.col
        og["REF"][0, :, 4] = REF5.data
        og["REF"][1, :, 4] = REF5.row
        og["REF"][2, :, 4] = REF5.col

        og["TR1"] = np.empty((3, TR1.nnz))
        og["TR1"].attrs["nnz"] = TR1.nnz
        og["TR1"].attrs["ds"] = ds1
        og["TR1"][0, :] = TR1.data
        og["TR1"][1, :] = TR1.row
        og["TR1"][2, :] = TR1.col

        og["TR2"] = np.empty((3, TR2.nnz))
        og["TR2"].attrs["nnz"] = TR2.nnz
        og["TR2"].attrs["ds"] = ds2
        og["TR2"][0, :] = TR2.data
        og["TR2"][1, :] = TR2.row
        og["TR2"][2, :] = TR2.col

        f.close()
