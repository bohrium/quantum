''' author: samtenka
    change: 2021-08-15
    create: 2021-08-14
'''

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt 

'''
--- SETUP

    Let's study multi-electron atoms via a mean field approximation for
    electron-electron interaction.  For a nucleus of charge Z surrounded by
    Z' electrons, we imagine that each electron sees a rotationally symmetric
    shielded Coulomb potential

      V(r) = (W(r)-Z)/r

    where W(r) is the average number of electrons (among the other Z'-1 many
    electrons) contained in a ball of radius r.  Note that for r << 1/Z, the
    electron density is close to constant so W(r) scales like   

    We'd like the energy levels of a shielded
    Coulomb potential V(r).  scales like V(r) ~ -Z/r for r<<1/Z and like V(r) ~
    -Z'/r for r>>1.  

    Applying a hydrogenic ansatz, we seek a wavefunction of the form 

      wavefunction = phi * psi = (f(r) * exp(- a r)) * psi(angle)

    whose angular component obeys (let NN be the spherical laplacian):

      E_a psi = -(1/2) NN psi 

    and whose radial component obeys (let D be the radial partial):

      E phi = H phi
            = -(1/2) (DD phi + 2 (D phi) / r) -- radial kinetic
              + (E_a/r^2 + V) phi             -- centifical + potential

--- ANGULAR COMPONENT

    We guess that psi is a polynomial p restricted to the unit sphere.  Due to
    the sphere's curvature, even linear polynomials are not harmonic.
    Evaluating in the unit direction +z, and writing X,Y,Z for the three
    cartesian partials, we have:

      NN p = (XX+YY) (p circ (1-sqrt(1+xx+yy),x,y))
           =   X (-Zp*x + Xp*1) 
             + Y (-Zp*y + Yp*1)
           = (XX+YY) p - 2 Zp 

    More generally, in the direction v=(x,y,z) we have:

      NN p = (XX+YY+ZZ) p - 2(xX+yY+zZ) p - (xX+yY+zZ)^2 p

    where 

      (*) the derivatives of (xX+yY+zZ)^2 do not act on its coefficients.

    For instance,

      (0) p = 1      --> NN p = 0 = -0p
      (1) p = z      --> NN p = -2z = -2p 
      (2) p = xy     --> NN p = -2(xy+yx) - 2xy = -6p         -- by (*)
      (3) p = zz-1/3 --> NN p = 2 - 4zz - 2zz = -6p

    As a check, we see by (3) that p = xx+yy+zz --> NN p = 6 - 6(xx+yy+zz) = 0
    on the unit sphere.  Good!  As a more complicated check, let's do a couple
    f orbitals (in each case, recall (*); for (5) note that 6*6/5=12*3/5):

      (4) p = xyz       --> NN s = -2(xyz+xyz+xyz) - 6xyz = -12p
      (5) p = z(zz-3/5) --> NN s = 6z - 2z(3zz-3/5) - zz(6z) = -12p

    The angular energies are 0, 1, 3, 6: the familiar sequence of triangular
    numbers.

--- RADIAL COMPONENT

    Reparameterize the radius to the interval [0,1] via

      s = exp(-r)           phi(r) = chi(exp(-r)) = chi(s)
      r = -log(s)           chi(s) = phi(-log(s)) = phi(r) 

    We check that

      (d/dr) phi   = -s (d/ds) chi 
      (d/dr)^2 phi = s^2 (d/ds)^2 chi + s (d/ds) chi

    Thus chi obeys the differential equation

      E chi =   (E_a/log(s)^2 + V(s)) chi
              - (s/2 + s/log(s)) (d/ds) chi
              - (s^2/2) (d/ds)^2 chi

    The solution for (n,l) should have

      phi(r) ~ C r^l                  for r << 1/Z
      phi(r) ~ C' r^n exp(-(Z'/n)r)   for r >> 1 
'''


Z = 1

dr = 0.0001 
max_r = 30
r    = np.arange(dr, max_r, dr)
idxs = np.arange(10, len(r)-10)

diff = lambda a: np.array(list((a[1:]-a[:-1])/dr) + [0]) 
potential = -Z/r 

for k, (enn, ell, phi) in enumerate([
                    (1, 0, np.exp(-r/1.0) * (  1                         )),
                    (2, 0, np.exp(-r/2.0) * (  2 -     r                 )),
                    (2, 1, np.exp(-r/2.0) * (          r                 )),
                    (3, 0, np.exp(-r/3.0) * ( 27 -  18*r +  2*r**2       )),
                    (3, 1, np.exp(-r/3.0) * (        6*r -    r**2       )),
                    (3, 2, np.exp(-r/3.0) * (                 r**2       )),
                    (4, 0, np.exp(-r/4.0) * (192 - 144*r + 24*r**2 - r**3)),
                    #(4, 1, np.exp(-r/4.0) * (       80*r - 20*r**2 + r**3)),
                    #(4, 2, np.exp(-r/4.0) * (              12*r**2 - r**3)),
                    #(4, 3, np.exp(-r/4.0) * (                        r**3)),
                    ]):
    color = ' rgby'[enn]
    offset = 0.08*k

    norm = np.sum((r*phi*phi)[idxs])
    distr = (r*phi*phi)[idxs]/norm
    quantiles = []
    rsum = 0.0
    goal = 0.0
    for k, v in enumerate(distr):
        rsum += v
        if rsum <= goal: continue
        goal += 1.0/200
        quantiles.append(r[idxs][k])
    quantiles = np.array(quantiles)

    plt.scatter(quantiles*0.0 + offset, quantiles, marker='_', alpha=0.05, color=color)

    #plt.plot(distr, r[idxs],
    #        linestyle='solid dashed dotted'.split()[ell],
    #        color=color,
    #        lw=0.8)

    amax = r[idxs][np.argmax(distr)]
    vmax = distr[np.argmax(distr)] 
    #amean = np.sum((r[idxs])*distr)
    plt.text(offset+0.03, amax,
            '{}{}'.format(enn, 'spdf'[ell]),
            ha='center', color=color, family='serif')
    #plt.text(offset, -1.0,
    #        '{}{}'.format(enn, 'spdf'[ell]),
    #        ha='center', color=color, family='serif')

    Dphi  = diff(phi)
    DDphi = diff(Dphi)
    
    E_a   = ell*(ell+1)/2.0 
    H_phi = (potential + E_a/r**2)*phi - 0.5*DDphi - Dphi/r

    energy = np.median(H_phi[idxs]/phi[idxs]) 
    print('energy for n={}, l={} is -1.0/{:5.2f}'.format(
        enn, ell, -1.0/energy))

plt.gcf().set_size_inches(2.0, 6.0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

#plt.gca().xaxis.set_ticks(np.arange(0,1+1,1))
#plt.gca().spines['bottom'].set_bounds(0.0, 1.0)
plt.gca().xaxis.set_ticks([])

plt.axis(ymin=max_r, ymax=0.0)
plt.gca().yaxis.set_ticks([0, 1] + list(range(5, max_r+1, 5)))
plt.gca().spines['left'].set_bounds(0.0, max_r)

plt.savefig('hi.png', bbox_inches='tight')
        


