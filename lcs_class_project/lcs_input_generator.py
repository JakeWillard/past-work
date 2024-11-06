'''
Double-Gyre LCS Code Input File Generator

This code creates an input file for the LCS code.  File is an hdf5 file that 
contains x and y velocity values over time for a grid

Variable paramenters are:
    time (in)dependence in velocity values
    grid spacing and size in x and y
    time duration and increment values

The velocity values are from the analytic solution of the double-gyre as described on 
https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/


Written by Kelly Cantwell
Last Updated 3/5/2022
'''

# Alex: commented out the line that flips y, that seems to mess things up

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import time


#### Desired File Paramenters ####

epsilon = 0.5 #how far r/l the oscilation moves
A = 0.15 #velocity amplitude
period = 5 # period of the oscillation

x_min = 0.5
x_max = 2.5
y_min = 0.5
y_max = 1.5

dx = 0.01
dy = 0.01

run_time = 20
dt = 0.05


identifier = 'lcs_example'

##################################




def init_file():
    
    if epsilon>0:
        dep = 'time_varying'
    elif epsilon==0:
        dep = 'stationary'
    
    file_name = 'double_gyre_velocity_fields_' + str(identifier) + '_' + str(dep) + '.h5'
    
    f = h5.File(file_name, "w")

    return f


def define_arrays():
    # create the x, y, t value arrays
    X = np.arange(x_min, x_max+dx, dx)
    Y = np.arange(y_min, y_max+dy, dy)
    T = np.arange(0, run_time+dt, dt)
    
    return X, Y, T

def match_arrays(X, Y, T):
    #Make arrays of the x, y, t values of the same shape as the velocity output
    x_2d = X
    # Y = np.flip(Y)
    y_2d = Y
    
    for i in range(len(X)-1):
        y_2d = np.vstack((y_2d, Y))
    y_2d = y_2d.transpose()
    for i in range(len(Y)-1):
        x_2d = np.vstack((x_2d, X))
        
    x = x_2d
    y = y_2d
    t_2d = np.full((len(Y), len(X)),T[0])
    t = t_2d
    for i in range(len(T)-1):
        x = np.dstack((x, x_2d))
        y = np.dstack((y, y_2d))
        
        t_2d = np.full((len(Y), len(X)),T[i+1])
        t = np.dstack((t, t_2d))
        
    return x,y,t


def calc_velocity(x, y, t):
    
    Vx = np.zeros_like(x)
    Vy = np.zeros_like(y)
       
    freq = 1/period
    w = 2*np.pi*freq
    
    #calcualte the velocities
    
    a = epsilon*np.sin(w*t)
    b = 1 - 2*epsilon*np.sin(w*t)
    
    f_xt = a*(x**2) + b*x
    
    Vy = -(np.pi)*A*np.sin(np.pi*f_xt)*np.cos(np.pi*y)
    Vx = np.pi*A*np.cos(np.pi*f_xt)*np.sin(np.pi*y)*((2*a*x) + b)
    
    
    return Vx, Vy



#### "Main Method" ####

X, Y, T = define_arrays()

x, y, t = match_arrays(X, Y, T)

Vx, Vy = calc_velocity(x, y, t)

# Making border of zeros
Vx[:,0,:] *= 0
Vx[:,-1,:] *= 0
Vx[0,:,:] *= 0
Vx[-1,:,:] *= 0
Vy[:,0,:] *= 0
Vy[:,-1,:] *= 0
Vy[0,:,:] *= 0
Vy[-1,:,:] *= 0

# fig, ax = plt.subplots()
# 
# for i in range(len(T)):
#     
#     xs = x[0,:,i]
#     ys = y[:,0,i]
#     Vxs = Vx[:,:,i]
#     Vys = Vy[:,:,i]
# 
#     Vxs = Vx[:,:,i]
#     Vys = Vy[:,:,i]
#     
#     ax.clear()
#     ax.set_title('Velocity Field')
#     
#     plt.quiver(xs, ys, Vxs, Vys)
#     plt.show()
#     
#     time.sleep(dt)
    


f = init_file()

dataset_X = f.create_dataset("X", data = X)
dataset_Y = f.create_dataset("Y", data = Y)

timestepNUM = f.create_group('timestepNUM')

dataset_Vx = timestepNUM.create_dataset("Vx", data = Vx)
dataset_Vy = timestepNUM.create_dataset("Vy", data = Vy)


f.close()
