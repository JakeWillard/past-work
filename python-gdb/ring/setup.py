
import h5py
import geomod
import numpy as np
import matplotlib.pyplot as plt


def plot_points(grid):

    cg = grid.points["core-ghost"]
    wg = grid.points["wall-ghost"]
    pg = grid.points["plate-ghost-pos-flux"]
    ng = grid.points["plate-ghost-neg-flux"]
    ip = grid.points["interior"]

    plt.scatter([p[1] for p in cg], [p[2] for p in cg], c="r")
    plt.scatter([p[1] for p in wg], [p[2] for p in wg], c="b")
    plt.scatter([p[1] for p in pg], [p[2] for p in pg], c="g")
    plt.scatter([p[1] for p in ng], [p[2] for p in ng], c="m")
    plt.scatter([p[1] for p in ip], [p[2] for p in ip], c="k")

    plt.show()


def setup_pde(f, grid, nz, dr, dt):

    dz = 2 * np.pi / nz

    REF, nnz_REF = grid.REF()
    PEN = grid.PEN(dr, dt)
    TR1 = grid.TR1(dz)
    TR2 = grid.TR2(dz)

    ds1 = (2) * np.ones(grid.N)
    ds2 = ds1

    px = grid.partial_x()
    pxx = grid.partial_xx()
    pxy = grid.partial_xy()
    py = grid.partial_y()
    pyy = grid.partial_yy()
    pc = -2 * py

    f["/pde/REF"] = REF
    f["/pde/nnz_REF"] = nnz_REF
    f["/pde/PEN"] = PEN
    f["/pde/TR1"] = TR1
    f["/pde/TR2"] = TR2
    f["/pde/ds1"] = ds1
    f["/pde/ds2"] = ds2
    f["/pde/partial_x"] = px
    f["/pde/partial_xx"] = pxx
    f["/pde/partial_xy"] = pxy
    f["/pde/partial_y"] = py
    f["/pde/partial_yy"] = pyy
    f["/pde/partial_c"] = pc


def main():

    N = 100
    xs = np.linspace(-1.1, 1.1, N)
    ys = np.linspace(-1.1, 1.1, N)
    geometry = geomod.RingGeometry(0.1, 0.1)
    grid = geomod.GDBGrid(geometry, xs, ys, 4, 4, 0.05, 0.05)

    # plot_points(grid)
    f = h5py.File("test.h5", "w")
    setup_pde(f, grid, 10, 0.05, 0.05)
    f.close()


if __name__ == "__main__":
    main()
