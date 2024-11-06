# mask off
import argh
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import h5py




# ===================================
# LAGRANGE CALCULATIONS
# ===================================


class Tracer:

    def __init__(self, U, x_coords, y_coords, dt):

        self.dt = dt
        self.vx = RectBivariateSpline(x_coords, y_coords, U[:,:,0])
        self.vy = RectBivariateSpline(x_coords, y_coords, U[:,:,1])

    def __call__(self, x, y, i, j):

        #xn, yn = rk4_trace(self.v, np.array([x,y]), self.dt, 10)

        f = lambda t,r: np.array([self.vx(r[0], r[1])[0][0], self.vy(r[0], r[1])[0][0]])
        xn, yn = solve_ivp(f, (0.0, self.dt), [x,y], t_eval=[self.dt]).y[:,0]

        return np.array([xn, yn, i, j])


def compute_phi(U, x_coords, y_coords, dt, nprocs):

    Nx, Ny, Nt, dummy = U.shape
    Phi = np.zeros(U.shape)

    # set the velocities near the boundary to zero
    U[:,0,:,:] *= 0
    U[:,-1,:,:] *= 0
    U[0,:,:,:] *= 0
    U[-1,:,:,:] *= 0

    with tqdm(total=Nx*Ny*Nt) as pbar:

        for t in range(0, Nt):

            # initialize tracer
            tr = Tracer(U[:,:,t,:], x_coords, y_coords, dt)

            with ProcessPoolExecutor(max_workers=nprocs) as exe:

                # create futures set
                futures = set()

                # submit tracing jobs and assemble set of futures
                for i in range(0, Nx):
                    for j in range(0, Ny):
                        x = x_coords[i]
                        y = y_coords[j]
                        futures.add(exe.submit(tr, x_coords[i], y_coords[j], i, j))

                # obtain results from futures as the traces are completed
                for fut in as_completed(futures):
                    x, y, i, j = fut.result()
                    Phi[int(i),int(j),t,:] = [x, y]
                    pbar.update(1)


    return Phi


# ===================================
# FTLE CALCULATIONS
# ===================================


def composite_phis(philist,rangex,rangey,interporder='cubic'):

    # Separating out components and changing indexes so we can iterate over time
    phixlist = np.moveaxis(philist[:,:,:,0],-1,0)
    phiylist = np.moveaxis(philist[:,:,:,1],-1,0)

    # Defining grid
    [ygrid,xgrid] = np.meshgrid(rangey,rangex)

    # The first phi is the identity map, aka the grid
    phixc = xgrid
    phiyc = ygrid

    # For plotting purposes, uncomment if you want to plot
    # Decimation factor for drawing field lines and arrows
    # decx = int(xgrid.shape[0]/20)
    # decy = int(xgrid.shape[1]/20)

    # Iteratively compositing the phis
    for i in range(phixlist.shape[0]):
        # Interpolating the ith phi
        if interporder=='cubic':
            phixinterp = RectBivariateSpline(rangex,rangey,phixlist[i],kx=3,ky=3)
            phiyinterp = RectBivariateSpline(rangex,rangey,phiylist[i],kx=3,ky=3)
        elif interporder=='linear':
            phixinterp = RectBivariateSpline(rangex,rangey,phixlist[i],kx=1,ky=1)
            phiyinterp = RectBivariateSpline(rangex,rangey,phiylist[i],kx=1,ky=1)
        else:
            raise Exception('only linear and cubic interpolation supported')

        # Compositing the ith phi with the previous composition of phi_0, ..., phi_{i-1}
        phixcnew = phixinterp(phixc,phiyc,grid=False)
        phiycnew = phiyinterp(phixc,phiyc,grid=False)

        # Optional: plotting. This traces out flow lines. It will significantly slow things down!
        # Comment or uncomment as desired
        # for j in range(len(xgrid[::decx,::decy].reshape(-1))):
        #     plt.plot([phixc[::decx,::decy].reshape(-1)[j],phixcnew[::decx,::decy].reshape(-1)[j]],[phiyc[::decx,::decy].reshape(-1)[j],phiycnew[::decx,::decy].reshape(-1)[j]],color='red')
        # if i == (phixlist.shape[0]-1):
        #     plt.quiver(xgrid[::decx,::decy],ygrid[::decx,::decy],(phix-xgrid)[::decx,::decy],(phiy-ygrid)[::decx,::decy])

        # Updating the composited phi to include the contribution of phi_i
        phixc = phixcnew
        phiyc = phiycnew

    # Creating the new phi
    phic = np.moveaxis(np.asarray([phixc,phiyc]),0,-1)

    # Checking to see what fraction of particles advected past the simulation boundary
    # This can be commented out with no issue
    ################################################################################################################################
    ################################################################################################################################
    boundaryx = [rangex[1],rangex[-2]]
    boundaryy = [rangey[1],rangey[-2]]
    phixtrim = phic[1:-1,1:-1,0]
    phiytrim = phic[1:-1,1:-1,1]
    nparticles = (len(rangex)-1)*(len(rangey)-1)

    oobx = ((phixtrim<boundaryx[0]) | (phixtrim>boundaryx[1]))
    ooby = ((phiytrim<boundaryy[0]) | (phiytrim>boundaryy[1]))
    oob = (oobx | ooby)

    nout = np.count_nonzero(oob)
    fracout = nout/nparticles

    #print('fraction of particles lost:')
    #print(fracout)
    ################################################################################################################################
    ################################################################################################################################


    return phic


def compute_FTLEs(phi,rangex,rangey,deltaT):
    # Breaking up phi into components

    phix = phi[:,:,0]
    phiy = phi[:,:,1]


    # Generating grid from x and y values
    [ygrid,xgrid] = np.meshgrid(rangey,rangex)

    # Getting grid shape
    gridshape = xgrid.shape

    # Grid of zeros, with the correct shape
    zerogrid = np.zeros(gridshape)

    # Calculating finite difference jacobian components (but with zeros on the boundary, since it's not defined there)
    Jxx = np.copy(zerogrid)
    Jxy = np.copy(zerogrid)
    Jyx = np.copy(zerogrid)
    Jyy = np.copy(zerogrid)

    # final separation in x for initial x separation of one x grid spacing
    Jxx[1:-1,1:-1] = (phix[2:,1:-1]-phix[:-2,1:-1])/(xgrid[2:,1:-1]-xgrid[:-2,1:-1])
    # final separation in y for initial x separation of one x grid spacing
    Jyx[1:-1,1:-1] = (phiy[2:,1:-1]-phiy[:-2,1:-1])/(xgrid[2:,1:-1]-xgrid[:-2,1:-1])
    # final separation in y for initial y separation of one y grid spacing
    Jyy[1:-1,1:-1] = (phiy[1:-1,2:]-phiy[1:-1,:-2])/(ygrid[1:-1,2:]-ygrid[1:-1,:-2])
    # final separation in x for initial y separation of one y grid spacing
    Jxy[1:-1,1:-1] = (phix[1:-1,2:]-phix[1:-1,:-2])/(ygrid[1:-1,2:]-ygrid[1:-1,:-2])

    # We flatten all the jacobian components into 1d arrays, arrange them into the order we want them in, then move the axes to form an array of 2x2 matrices
    # We can later reshape back to the grid shape to recover the original indexing
    J = np.moveaxis(np.asarray([[Jxx.reshape(-1), Jxy.reshape(-1)],[Jyx.reshape(-1),Jyy.reshape(-1)]]),-1,0)

    # Taking singular values, but not the corresponding vectors, and separating into max and mins
    svdmax,svdmin = np.linalg.svd(J,compute_uv=False).T

    # If this condition is true, we have found a source or a sink. They are masked so they aren't mistakenly characterized as
    # part of material lines
    # This has a side effect of masking the boundary where the jacobians are all zero, which is nice.
    badcondition = ((svdmax < 1) | (svdmin > 1))

    # No masking!!!!
    ## The properly shaped masked array of maximum singular values, corresponding to the grid
    # svdmat = np.ma.masked_where(badcondition, svdmax).reshape(gridshape)
    svdmat = svdmax.reshape(gridshape)

    # The finite time lyapunov exponents
    ftles = np.log(svdmat)/np.abs(deltaT)

    # Optional: plotting
    # Comment or uncomment as desired

    # decx=int(gridshape[0]/10)
    # decy=int(gridshape[1]/10)
    #
    # plt.pcolormesh(xgrid[1:-1,1:-1],ygrid[1:-1,1:-1],np.log((svdmat[1:-1,1:-1])),shading='auto')
    # plt.colorbar()
    # plt.quiver(xgrid[::decx,::decy],ygrid[::decx,::decy],(phix-xgrid)[::decx,::decy],(phiy-ygrid)[::decx,::decy],color='red')

    return ftles


# ===================================
# READ INPUT DATA
# ===================================


def extract_GameraData(filename):
    """
    Decscription: Extract Gamera data and reshape grid.
    NOTE: Does not output the input data to the compute_phi.py script. (see import_GameraData function)
    Top Levels of f:
    * 'Step#<timestepNUM>': Subgroup containing data at timestep timestepNUN
        * 'D' : density (shape =n-1xm-1)
        * 'Vx' : Velocity X-Component (shape =n-1xm-1)
        * 'Vy' : Velocity Y-Component (shape =n-1xm-1)
    * 'X': nxm Array containing the n X values in the xy spatial grid (values only vary in the X[i, :])
    * 'Y': nxm Array containing the m Y values in the xy spatial grid (values only vary in the Y[:, i])
    * 'dV': Array of values

    Output:
        Vx_arr : n-1xm-1xk array containinng the Vx data for each of the k time steps at the center of each of the n X values and m Y values.
        Vy_arr : n-1xm-1xk array containinng the Vy data for each of the k time steps at the center of each of the n X values and m Y values.
    """
    scratch_path = '/glade/scratch/mlmoses/'
    run_prmDct = {scratch_path+'test/Hall8.gam.h5': {'dt':0.5, 'tFin':60.0},
            scratch_path+'McNally1/McNally1.gam.h5': {'dt':0.5, 'tFin':60.0}, # McNally with the same rectangular grid and boundary conditions as Hall8 case
            scratch_path+'McNally_simple/McNally0.gam.h5': {'dt':0.01, 'tFin':5.0}, # Stripped Down McNally from Prof Example with a square grid
            scratch_path+'McNally0_dbGyrBnds/McNally0b.gam.h5': {'dt':0.1, 'tFin':10.0}, # Stripped Down McNally with same spatial and temporal grid as Double Gyre time varying file.
            scratch_path+'KH1/Sim.gam.h5': {'dt':0.5, 'tFin':60.0}}
    dt = run_prmDct[filename]['dt']
    time_arr = np.arange(0, run_prmDct[filename]['tFin']+dt, dt)
    f = h5py.File(filename, 'r')
    X = f['X'][()].T
    Y = f['Y'][()].T
    Vx_arr = np.zeros((len(np.unique(X))-1, len(np.unique(Y))-1, len(time_arr)))
    Vy_arr = np.zeros((len(np.unique(X))-1, len(np.unique(Y))-1, len(time_arr)))
    for stepNum in np.arange(0, len(time_arr)):
        print(stepNum)
        stepT = 'Step#%s' % (stepNum)
        print(stepT)
        Vx_arr[:, :, stepNum] = f[stepT+'/Vx'][()].T # Velocity X-Component
        Vy_arr[:, :, stepNum] = f[stepT+'/Vy'][()].T # Velocity Y-Component
    return dt, time_arr, X, Y, Vx_arr, Vy_arr

def import_GameraData(filename):
    """
    Import Gamera Data into the compute_phi.py script.
    Note: This function calls the extract_GameraData to get the Gamera Output grids (reshaped).
    """
    dt, time_arr, X, Y, Vx_arr, Vy_arr = extract_GameraData(filename)
    U = np.zeros((len(np.unique(X))-1, len(np.unique(Y))-1, len(time_arr), 2))
    U[:, :, :, 0] = Vx_arr
    U[:, :, :, 1] = Vy_arr
    x_coords = X[0:-1, 0]
    y_coords = Y[0, 0:-1]
    x_coords = x_coords + np.diff(X[:, 0])/2.
    y_coords = y_coords + np.diff(Y[0, :])/2.
    return dt, x_coords, y_coords, time_arr, U


def import_double_gyre(filename):
    """
    Read a Double Gyre .h5 data file to test code.
    """
    f = h5py.File(filename, 'r')
    x_coords = f['X'][()]
    y_coords = f['Y'][()]
    dt = 0.1
    Vx_arr = f['timestepNUM']['Vx'][()]
    Vy_arr = f['timestepNUM']['Vy'][()]

#    time_arr = np.arange(0, dt*Vx_arr.shape[-1], dt)
    U = np.zeros((len(x_coords), len(y_coords), Vx_arr.shape[-1], 2))
#    U = np.zeros((len(y_coords), len(x_coords), Vx_arr.shape[-1], 2))
    time_arr = []
    for inx in np.arange(0, Vx_arr.shape[-1],1):
        U[:, :, inx, 0] = Vx_arr[:, :, inx].T
        U[:, :, inx, 1] = Vy_arr[:, :, inx].T
        if inx == 0: time_arr.append(0)
        else: time_arr.append(time_arr[-1]+dt)

    return dt, x_coords, y_coords, np.array(time_arr), U



# ===================================
# COMMAND LINE INTERFACE
# ===================================


def lagrange(input_file, output_file, nprocs=1):

    if input_file[-6:-3] == 'gam': dt, x_coords, y_coords, t_coords, U = import_GameraData(input_file)
    else: dt, x_coords, y_coords, t_coords, U = import_double_gyre(input_file)

    # compute phi
    phi = compute_phi(U, x_coords, y_coords, dt, nprocs)

    # output phi, x_coords, y_coords, dt
    f = h5py.File(output_file, "w")
    dataset_X = f.create_dataset("X", data = x_coords)
    dataset_Y = f.create_dataset("Y", data = y_coords)
    dataset_t = f.create_dataset("T", data = t_coords)
    dataset_Phi = f.create_dataset("Phi", data = phi)
    f.close()


def ftle(input_file, output_file, nc=1):

    # read input
    f = h5py.File(input_file, "r")
    x_coords = f['X'][()].T
    y_coords = f['Y'][()].T
    t_coords = f['T'][()].T
    Phi = f['Phi'][()]
    dt = t_coords[1] - t_coords[0]
    f.close()

    # compute forward ftles
    Nx, Ny, Nt, dummy = Phi.shape
    phic = np.zeros((Nx, Ny, Nt-nc, 2))
    ftles = np.zeros((Nx, Ny, Nt-nc))
    with tqdm(total=(Nt-nc)*2) as pbar:
        for i in range(Nt-nc):
            phic[:,:,i,:] = composite_phis(Phi[:,:,i:(i+nc+1),:],x_coords,y_coords)
            pbar.update(1)
        for i in range(Nt-nc):
            ftles[:,:,i] = compute_FTLEs(phic[:,:,i,:],x_coords,y_coords,dt)
            pbar.update(1)

    # write output
    # output phi, x_coords, y_coords, dt
    f = h5py.File(output_file, "w")
    dataset_X = f.create_dataset("X", data = x_coords)
    dataset_Y = f.create_dataset("Y", data = y_coords)
    dataset_t = f.create_dataset("T", data = t_coords)
    dataset_Phi = f.create_dataset("FTLE", data = ftles)
    f.close()


def movie(filename, moviename, framerate=6):

    # read input
    f = h5py.File(filename, "r")
    x_coords = f['X'][()].T
    y_coords = f['Y'][()].T
    t_coords = f['T'][()].T
    ftles = f['FTLE'][()]
    dt = t_coords[1] - t_coords[0]
    f.close()

    # make new directory for frames
    os.mkdir("./frames")

    # output frames
    Nt = ftles.shape[2]
    ygrid,xgrid = np.meshgrid(y_coords,x_coords)
    with tqdm(total=Nt) as pbar:
        for i in range(Nt):

            # plt.pcolormesh(ftles[:,:,i])
            plt.clf()
            plt.pcolormesh(xgrid,ygrid,ftles[:,:,i])
            plt.colorbar()
            plt.savefig("./frames/frame{0}.png" .format(i))
            pbar.update(1)

    # call ffmpeg
    cmd = "ffmpeg -f image2 -framerate {0} -i ./frames/frame%d.png -c:v libx264 -pix_fmt yuv420p -crf 23 {1}.mp4".format(framerate, moviename)
    os.system(cmd)

    # delete frames
    os.system("rm -r ./frames/")


def whole_shebang(input_file, outputs_name, procs=1, nc=1, framerate=6):

    phi_file = outputs_name + "_phi.h5"
    ftle_file = outputs_name + "_ftle.h5"

    # call lagrange
    print("calculating phi...")
    lagrange(input_file, phi_file, nprocs=procs)

    # call ftle
    print("calculating ftles...")
    ftle(phi_file, ftle_file, nc=nc)

    # make movie
    print("making movie...")
    movie(ftle_file, outputs_name, framerate=framerate)





parser = argh.ArghParser()
parser.add_commands([lagrange, ftle, movie, whole_shebang])
if __name__ == '__main__':

    parser.dispatch()
