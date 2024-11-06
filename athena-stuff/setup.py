import os
import numpy as np


input_text = """
<comment>
problem   = reconnection
author    = Xiaocan Li
configure = python configure.py --prob reconnection -b -g --flux hlle --coord gr_user

<job>
problem_id      = {name}  # problem ID: basename of output filenames

<output1>
file_type       = vtk          #
variable        = prim          # prim or cons
id              = prim          # file identifier
dt              = {dt}           # time increment between outputs

<output2>
file_type       = rst           # Restart dump
dt              = 1.0           # time increment between outputs

<time>
cfl_number      = 0.2           # The Courant, Friedrichs, & Lewy (CFL) Number
nlim            = -1            # cycle limit
tlim            = {tlim}          # time limit
integrator      = vl2           # time integration algorithm
xorder          = 2             # order of spatial reconstruction
ncycle_out      = 1             # interval for stdout summary info

<mesh>
nx1             = 500          # Number of zones in X1-direction
x1min           = 0.0           # minimum value of X1
x1max           = 2.0           # maximum value of X1
ix1_bc          = periodic      # inner-X1 boundary flag
ox1_bc          = periodic      # outer-X1 boundary flag

nx2             = 500          # Number of zones in X2-direction
x2min           = 0.0           # minimum value of X2
x2max           = 2.0           # maximum value of X2
ix2_bc          = periodic      # inner-X2 boundary flag
ox2_bc          = periodic      # outer-X2 boundary flag

nx3             = 1             # Number of zones in X3-direction
x3min           = -0.5          # minimum value of X3
x3max           = 0.5           # maximum value of X3
ix3_bc          = periodic      # inner-X3 boundary flag
ox3_bc          = periodic      # outer-X3 boundary flag

num_threads = 1         # Number of OpenMP threads per process
#refinement = static
#refinement     = adaptive
#numlevel       = 3
#deref_count    = 1

<meshblock>
nx1        = 500        # Number of zones per MeshBlock in X1-direction
nx2        = 500        # Number of zones per MeshBlock in X2-direction
nx3        = 1         # Number of zones per MeshBlock in X3-direction

<hydro>
gamma = 1.6666666666666667      # gamma = C_p/C_v
iso_sound_speed = 1.0           # isothermal sound speed
sigma_max       = 0             # ceiling on 2pmag/ρ; only used if positive
beta_min        = 0.001         # floor on pgas/pmag; only used if positive
gamma_max       = 10            # ceiling on Lorentz factor
dfloor          = 1.0e-4        # Density floor
pfloor          = 1.0e-4        # Pressure floor

<coord>
a = 0.0
m = 0.0
K0 = {K0}
K1 = {K1}

<problem>
beta0           = 0.1           # initial plasma beta
vin_pert        = 0.1           # initial velocity perturbation
random_vpert    = 1             # whether velocity perturbation is random
eta_ohm         = 0.0        # Ohmic diffusion (Dimensionless Rm = 1/eta_Ohm)
forcefree       = 1             # forcefree flag (integer:0,1)
cs_width        = 0.005          # current sheet width
Bguide          = 0.0           # guide field / reconnecting component
num_cs          = 2             # number of current sheets
phi_pert        = 1.0E-3        # magnetic flux perturbation
pres_balance    = 1             # whether to balance the initial total pressure
uniform_rho     = 0             # whether the initial density is uniform
pert_B          = 1             # whether to perturb magnetic field
pert_V          = 0             # whether to perturb velocity field
set_d_floor     = 0             # whether to set density floor
d_floor         = 1.0e-2        # density floor
b0              = {b0}           # reconnection magnetic field strength
rho             = 1.0           # gas density
pgas            = 0.1           # gas pressure
"""


def save_input_file(name, K0, K1, dt, tlim, sigma):

    path = "./data/{name}/athinput.reconnection".format(name=name)

    with open(path, "w") as f:
        f.write(input_text.format(name=name, K0=K0, K1=K1, dt=dt, tlim=tlim, b0=np.sqrt(sigma)))
    

def save_metric_input(name, K0, K1):

    path = "./data/{name}/metric.coef".format(name=name)

    with open(path, "w") as f:
        f.write("a, b".format(a=K0, b=K1))
    

def save_video_script(name):

    path = "./data/{name}/make-video".format(name=name)
    
    with open(path, "w") as f:
        f.write("ffmpeg -pattern_type glob -i '*.png' {name}.mp4".format(name=name))


def save_run_script(name):

    d_path = "./data/{name}".format(name=name)
    i_path = "./data/{name}/athinput.reconnection".format(name=name)

    with open("./run-{name}".format(name=name), "w") as f:
        f.write("./athena/bin/athena -d {d} -i {i}".format(d=d_path, i=i_path))


if __name__ == "__main__":

    name = "curved_laminar"
    dt = 0.05
    tlim = 2.0
    K0 = 0.2
    K1 = 0.2
    sigma = 0.1

    os.mkdir("./data/{name}".format(name=name))

    save_input_file(name, K0, K1, dt, tlim, sigma)
    save_metric_input(name, K0, K1)
    save_video_script(name)
    save_run_script(name)

