# -develop LCS finder code



'''
3 Files:
    
    Vector field input file 
    
    VF gets the Phi calculated
    
    Phi file output
    
    PF gets LCS calculated
    
    Matrix or graphic output of LCS


'''




'''
Params:
    time step




take in the vector field input

assign Lagrange test points (grid pts)

Integrate over a time

Find the Jacobian (Nearest neighbor calc)

Find singular values

Create array of largest SVs for each initial point




'''




'''
File 1: Vector field input

generated from MHD code or analytic solution for double-gyre

filtetype: unknown

array of some dimension and length
'''



'''
File 2: Calculation of Phi.py

take in the vector field input
parse time steps of vector field

For each time step:
    define Lagrange test points (grid pts) 
    add resolution if desired
    
    
    **Parallel Computation**
    store vals
    integrate forwards
    store vals
    integrate backwards
    store vals


Output hdf5 file of backmaps, forwardmaps
'''



'''
File 3: Phi time maps

filetype: hdf5

arrays of backmap, forwardmap for each u grid point, copy of grid
'''



'''
File 4: LCS Calculation.py

Input the integration maps and grid

For forward maps:
    For each time step:
        Find the Jacobian (Nearest neighbor calc)
        Find singular values
        store largest
        
For backward maps:
    For each time step:
        Find the Jacobian (Nearest neighbor calc)
        Find singular values
        store largest
        
Create array of largest SVs for each grid point, time step, and direction
'''



'''
File 5: LCS Output

filetype: hdf5 and/or mp4
    
arrays of SVs or graphic animation output
    
    
'''
