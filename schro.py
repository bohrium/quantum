''' author: samtenka
    change: 2021-08-15
    create: 2021-08-14
'''

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import matplotlib.pyplot as plt 

Z = 1

dr = 0.0001 
max_r = 20
r    = np.arange(dr, max_r, dr)
idxs = np.arange(10, len(r)-10)

diff = lambda a: np.array(list((a[1:]-a[:-1])/dr) + [0]) 
potential = -Z/r 

for enn, ell, phi in [
                    (1, 0, np.exp(-r/1.0)                   ),
                    (2, 0, np.exp(-r/2.0) * (2.0-r)         ),
                    (2, 1, np.exp(-r/2.0) * r               ),
                    (3, 0, np.exp(-r/3.0) * (27-18*r+2*r**2)),
                    (3, 1, np.exp(-r/3.0) * (6*r-r**2)      ),
                    (3, 2, np.exp(-r/3.0) * r**2            ),
                    ][::-1]:
    color = ' rgb'[enn]

    norm = np.mean((r*phi*phi)[idxs]) * (len(r)*dr) 
    distr = (r*phi*phi)[idxs]/norm
    plt.plot(r[idxs], distr,
            linestyle='solid dashed dotted'.split()[ell],
            color=color,
            lw=0.8)

    amax = r[idxs][np.argmax(distr)]
    vmax = distr[np.argmax(distr)] 
    plt.text(amax+0.05, vmax+0.01,
            '{}{}'.format(enn, 'spdf'[ell]),
            ha='center', color=color, family='serif')

    Dphi  = diff(phi)
    DDphi = diff(Dphi)
    
    E_a   = ell*(ell+1)/2.0 
    H_phi = (potential + E_a/r**2)*phi - 0.5*DDphi - Dphi/r

    energy = np.median(H_phi[idxs]/phi[idxs]) 
    print('energy for n={}, l={} is -1.0/{:5.2f}'.format(
        enn, ell, -1.0/energy))

plt.gcf().set_size_inches(8.0, 5.0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

#plt.gca().yaxis.set_ticks(np.arange(0,1+1,1))
#plt.gca().spines['left'].set_bounds(0.0, 1.0)
plt.gca().yaxis.set_ticks([])

plt.axis(xmin=0.0, xmax=max_r)
plt.gca().xaxis.set_ticks(np.arange(0,max_r+1,5))
plt.gca().spines['bottom'].set_bounds(0.0, max_r)

plt.savefig('hi.png', bbox_inches='tight')
        

'''

--- SETUP

    Let's visualize the subshells of a Coulomb + perturbation potential

      V(r) = -Z/r + q(r)

    Applying a hydrogenic ansatz, we seek a wavefunction of the form 

      wavefunction = phi * psi = (p(r) * exp(- a r)) * psi(angle)

    whose angular component obeys (let NN be the spherical laplacian):

      E_a psi = -(1/2) NN psi  

    and whose radial component obeys (let D be the radial partial):

      E_r phi = H phi
              = -(1/2) (DD phi + 2 (D phi) / r) -- radial kinetic
                + (E_a/r^2 + V) phi             -- centifical + potential

--- ANGULAR COMPONENT

    We guess that psi is a polynomial s restricted to the unit sphere.  Due to
    the sphere's curvature, even linear polynomials are not harmonic.
    Evaluating in the unit direction +z, and writing X,Y,Z for the three
    cartesian partials, we have:

      NN s = (XX+YY) (s circ (1-sqrt(1+xx+yy),x,y))
           =   X (-Zs*x + Xs*1) 
             + Y (-Zs*y + Ys*1)
           = (XX+YY) s - 2 Zs 

    More generally, in the direction v=(x,y,z) we have:

      NN s = (XX+YY+ZZ) s - 2(xX+yY+zZ) s - (xX+yY+zZ)^2 s

    where 

      (*) the derivatives of (xX+yY+zZ)^2 do not act on its coefficients.

    For instance,

      (0) s = 1      --> NN s = 0 = -0s
      (1) s = z      --> NN s = -2z = -2s 
      (2) s = xy     --> NN s = -2(xy+yx) - 2xy = -6s         -- by (*)
      (3) s = zz-1/3 --> NN s = 2 - 4zz - 2zz = -6s

    As a check, we see by (3) that s = xx+yy+zz --> NN s = 6 - 6(xx+yy+zz) = 0
    on the unit sphere.  Good!  As a more complicated check, let's do a couple
    f orbitals (in each case, recall (*); for (5) note that 6*6/5=12*3/5):

      (4) s = xyz       --> NN s = -2(xyz+xyz+xyz) - 6xyz = -12s
      (5) s = z(zz-3/5) --> NN s = 6z - 2z(3zz-3/5) - zz(6z) = -12s

    The angular energies are 0, 1, 3, 6: the familiar sequence l(l+1). 

--- RADIAL COMPONENT

'''
