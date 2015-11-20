import sys
import numpy as np

class Lattice(object):
  def __init__(self):
#  Using Silicon parameters:
#  lattice parameter of silicon (5.43 A)
    self.alat = 10.26
#  without the factor 4 we have the volume of 8 atoms...
    self.omega = 1080.49/4.0
#  !direct lattice vectors (cart. coord. in units of a_0)
#  ![Yu and Cardona, pag 23]
    self.at = np.array([[0.0,  0.5,  0.5 ], # ! a1
                        [0.5,  0.0,  0.5 ], # ! a2
                        [0.5, 0.5, 0.0]])   # ! a3
#  !reciprocal lattice vectors (cart. coord. in units 2 pi/a_0)
    self.bg = np.array([[-1.0,  1.0,  1.0],  # ! b1
                        [ 1.0, -1.0,  1.0],  # ! b2
                        [ 1.0,  1.0, -1.0]]) # ! b3

#  !atomic coordinates (cart. coord. in units of a_0)
    self.tau   = np.array([[ 0.125,  0.125,  0.125],
                           [-0.125, -0.125, -0.125]])
    self.tpiba2 = (2.0*np.pi/self.alat)**2
    self.ecut   = 4.0 
    self.gcut   = (4.0*self.ecut)/self.tpiba2

class Hamiltonian(object):
  def __init__(self):
    self.system = 'si'

  def build_pseud_pot(self):
#  diamond strubture - only the symmetric form factor is nonzero, Ry
#  from: Cohen & Bergstresser, PRB 141, 789 (1966)
    v3  = -0.21
    v8  = 0.04 
    v11 = 0.08 

  def h_psi(**kwargs):
    pass

  def eigenstates(self):
    vggp = self.build_pseud_pot()
    return electrons

class WaveFunction(object):
  def __init__(self):
    self.ecut  = 3.00
    self.npw   = self.ecut
    self.gvecs = np.zeros([self.npw])
    self.kvec  = np.zeros(3)
    self.lattice = Lattice()
    self.gcutm   = (4.0*self.ecut)/self.lattice.tpiba2
# realspace grid
    self.nr1   = 1 + int (2 * np.sqrt(self.gcutm)*np.sqrt(sum(self.lattice.at[:,0]*self.lattice.at[:,0])))
    self.nr2   = 1 + int (2 * np.sqrt(self.gcutm)*np.sqrt(sum(self.lattice.at[:,1]*self.lattice.at[:,1])))
    self.nr3   = 1 + int (2 * np.sqrt(self.gcutm)*np.sqrt(sum(self.lattice.at[:,2]*self.lattice.at[:,2])))
    self.nrx   = self.nr1*self.nr2*self.nr3

  def gen_field(self):
     return meshgrid(2*self.nr1+1, 2*self.nr2+1, 2*self.nr3+1)

  def fft(self, type='fw'):
    if type=='fw':
       np.fft.fftn  (self.coeff)
    else:
       np.fft.ifftn (self.coeff)

  def sample_freq(self):
    np.fft.fftfreq(self.nr1)

  def laplacian3d(self, field, KX, KY, KZ):
    return [ifft(-KX**2*fft(field, axis = 0), axis = 0) + 
           ifft(-KY**2*fft(field, axis = 1), axis = 1) + 
           ifft(-KZ**2*fft(field, axis = 2), axis = 2) ]

  def v_pot(self, vr):
    return vr*self.gen_field()

class GradientSolvers(object):
  def __init__(self):
    self.maxter = 120

  def cg_solve(self):
    pass

  def cbicg_solve(self):
    pass

  def cbcg_multi_from(self, A, b, eigs, nbnd=1, ndim=50, nfreq=1):
    alpha       = 1.0
    beta        = 1.0
    rho_old     = np.ones(nbnd)
    rho_sig_old = np.ones(nbnd)
    pi_sig_old  = np.ones(nbnd)
    pi_sig      = np.ones(nbnd)
    g           = np.zeros(nbnd, ndim)
    gt          = np.zeros(nbnd, ndim)
    h           = np.zeros(nbnd, ndim)
    ht          = np.zeros(nbnd, ndim)
    u           = np.zeros(nbnd, ndim)
    u_til       = np.zeros(nbnd, ndim)
    u_sig       = np.zeros(nbnd, nfreq, ndim)
    a           = np.zeros(nbnd)
    c           = np.zeros(nbnd)
    conv        = np.zeros(nbnd)

    h_psi = Hamiltonian.h_psi()

    for step in range(1, self.maxter+1):
      if step==1:
        for iw in range(nfreq):
          u_sig[:,iw,:] = b[:, :]
        g[:,:]  = b[:,:]
        gt[:,:] = np.conj(b[:,:])
        ht[:,:] = np.conj(b[:,:])

        alpha_old   = np.zeros(nbnd)
        beta_old    = np.zeros(nbnd)
        pi_sig_old  = np.ones(nbnd)
        pi_sig      = np.ones(nbnd)

    if conv_root break  
# APPLY HAMILTONIAN TO SEARCH DIRECTION      
      h_psi(ndim, nbnd, nfreq, t,  eigs)
      h_psi(ndim, nbnd, nfreq, tt, eigs)
    
    lbnd = 0
    for ibnd in range(nbnd):
      if conv[ibnd]==0:
        lbnd += 1
        a[ibnd] = np.vdot(gt[:,ibnd], g[:,ibnd])
        c[ibnd] = np.vdot(ht[:,ibnd], t[:,lbnd])

    lbnd = 0
    for ibnd in range(nbnd):
      if conv[ibnd]==0:
      lbnd += 1
        for iw in range(nfreq):
          pi_sig_new[ibnd, iw] = 
      


  return x_sig

if __name__=='__main__':
  psi = WaveFunction()
  print psi.lattice.alat, psi.lattice.omega, psi.nr1, psi.nr2, psi.nr3, psi.nrx


