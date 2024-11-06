
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from multiprocessing import Pool
import numpy as np
import sys
import read_GameraData as rdGam
# import matplotlib.pyplot as plt


# combine flow field data into a callable object for tracing.
class Tracer:

    def __init__(self, U, x_coords, y_coords, dt):

        self.dt = dt
        self.interp_x = RectBivariateSpline(x_coords, y_coords, U[:,:,0])
        self.interp_y = RectBivariateSpline(x_coords, y_coords, U[:,:,1])

    def __call__(self, x, y):
        return solve_ivp(lambda t, r: [self.interp_x(r[0], r[1])[0][0], self.interp_y(r[0], r[1])[0][0]], (0.0, self.dt), [x, y], t_eval=[self.dt]).y[:,0]


if __name__ == '__main__':

#    dt, x_coords, y_coords, time_arr, U = rdGam.import_GameraData(str(sys.argv[-1]))
    dt, x_coords, y_coords, time_arr, U = rdGam.import_double_gyre(str(sys.argv[-1]))
    Nx, Ny, Nt = (len(x_coords), len(y_coords), len(time_arr))
#    # placeholder assignments (these will be read in from file)
#    dt = 0.01
#    Nx = 20
#    Ny = 0
#    Nt = 1
#    x_coords = np.linspace(0, 1, Nx)
#    y_coords = np.linspace(0, 1, Ny)
#    U = np.zeros((Nx, Ny, Nt, 2))

    # initialize Phi
    Phi = np.zeros((Nx, Ny, Nt, 2))

    for t in range(0, Nt):

        # TODO: Parallelize the (i,j) iterations
        tr = Tracer(U[:,:,t,:], x_coords, y_coords, dt)
        for i in range(0, Nx):
            for j in range(0, Ny):
                Phi[i,j,t,:] = tr(x_coords[i], y_coords[j])


    # save Phi to file
    
    # A bit of fake code to explain how to use the functions:
    
    # Let's say that we decided that a finite time integration of 7 timesteps is a good amount
    # Let one timestep be dT.
    # The forward integration code would look like this:
    
    Phi_7timesteps = np.zeros((Nx, Ny, Nt, 2))
    ftlesforward = np.zeros((Nx, Ny, Nt))
    ftlesbackward = np.zeros((Nx, Ny, Nt))
    for i in range(Nt-7):
        Phi_7timesteps[:,:,i,:] = composite_phis(Phi[:,:,i:(i+7+1),:],x_coords,y_coords)
    for i in range(Nt-7):
        ftlesforward[:,:,i] = compute_FTLES(Phi_7timesteps[:,:,i,:],x_coords,y_coords,dT)
    # We could combine the loops if we wanted ofc
    
    # The backwards integration code would look like
    
    Phi_reverse_7timesteps = np.zeros((Nx, Ny, Nt, 2))
    ftles = np.zeros((Nx, Ny, Nt))
    for i in range(Nt-7):
        Phi_reverse_7timesteps[:,:,i+7,:] = composite_phis(Phi_reverse[:,:,i:(i+7+1),:][:,:,::-1,:],x_coords,y_coords)
    for i in range(Nt-7):
        ftlesbackward[:,:,i] = compute_FTLES(Phi_reverse_7timesteps[:,:,i,:],x_coords,y_coords,-dT)



