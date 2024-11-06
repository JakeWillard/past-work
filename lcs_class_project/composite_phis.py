import numpy as np
from scipy.interpolate import RectBivariateSpline
# import matplotlib.pyplot as plt


# Composits a set of Phis in the order that they're given to it
# Takes in: philist, rangex, rangey, order
# The philist is a numpy array, indexed like [initial x coord of test particle, initial y coord of test particle, time, map component]
# If they are positive-time phis, they are already sorted in the right order. If they are negative time then the order must be reversed,
# for example by doing Phi[:,:,::-1,:]
# rangex: the ordered set of x values of points on the grid
# rangey: the prdered set of y values of points on the grid
# interporder: what order interpolation to use. Default is cubic, you can also use linear.

# Returns a new phi that is the composition of the given phis

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
    
    print('fraction of particles lost:')
    print(fracout)
    ################################################################################################################################
    ################################################################################################################################

    
    return phic
