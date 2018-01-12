import pyscf
import numpy
import pyfftw
import time
import ase
import sys
import fft_cython

from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import df as pbcdf
from pyscf.pbc import tools as pbctools
from pyscf import lib as pylib
from pyscf.pbc.dft import numint
from pyscf.lib import logger
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from pyscf.pbc.dft.gen_grid import gen_uniform_grids

pyfftw.interfaces.cache.enable()

def get_cell(lc_bohr,atom,unit_cell,basis,mesh,pseudo,supercell):

    cell=pbcgto.Cell()
    boxlen=lc_bohr
    ase_atom=ase.build.bulk(atom, unit_cell, a=boxlen)
    cell.a=ase_atom.cell
    cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.basis=basis
    cell.charge=0
    cell.dimension=3
    cell.incore_anyway=False
    cell.max_memory=8000
    cell.mesh=numpy.array([mesh,mesh,mesh])
    cell.pseudo=pseudo
    cell.spin=0
    cell.unit='B'
    cell.verbose=10
    print "*****BUILDING CELL!*****"
    cell.build()

    if supercell!=[]:
        print "*****BUILDING SUPERCELL!*****"
        cell._built=False
        cell=pbctools.super_cell(cell, supercell)

    return cell

def get_scf(cell):

    scf=pbchf.RHF(cell, exxdiv=None)
    scf.conv_tol=1e-10
    scf.conv_tol_grad=1e-8
    scf.diis=True
    scf.diis_space=20
    scf.direct_scf_tol=1e-14
    scf.init_guess='minao'
    scf.max_cycle=200
    scf.max_memory=8000
    scf.verbose=10
    scf.scf()

    return scf

def get_nocc(mp):
    if mp._nocc is not None:
        return mp._nocc
    else:
        nocc = numpy.count_nonzero(mp.mo_occ > 0)
        assert(nocc > 0)
        return nocc

def get_nmo(mp):
    if mp._nmo is not None:
        return mp._nmo
    else:
        return len(mp.mo_occ)

def get_ngs(mp):
    if mp._ngs is not None:
        return mp._ngs
    else:
        return numpy.prod(mp._scf.cell.mesh)

class LTSOSMP2(pylib.StreamObject):
    def __init__(self, scf, mo_coeff=None, mo_occ=None):

        if mo_coeff  is None: mo_coeff  = scf.mo_coeff
        if mo_occ    is None: mo_occ    = scf.mo_occ

        self.mol = scf.mol
        self._scf = scf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = scf.max_memory
        self.optimization = 'Cython'
        self.lt_points = 7

##################################################
# don't modify the following attributes, they are not input options
        self.mo_energy = scf.mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self._ngs = None
        self.e_corr = None
        self._keys = set(self.__dict__.keys())

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    @property
    def ngs(self):
        return self.get_ngs()
    @ngs.setter
    def ngs(self, n):
        self._ngs = n

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s flags ********', self.__class__)
        log.info('nocc = %s, nmo = %s, ngs = %s', self.nocc, self.nmo, self.ngs)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, pylib.current_memory()[0])

    @property
    def emp2(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.e_corr + self._scf.e_tot

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_ngs = get_ngs

    def kernel(self, mo_energy=None, mo_coeff=None):

        if mo_energy is not None: self.mo_energy = mo_energy
        if mo_coeff is not None: self.mo_coeff = mo_coeff
        if self.mo_energy is None or self.mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call scf.kernel() to generate them.')
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_corr = kernel(self, mo_energy, mo_coeff, self.verbose)
        logger.log(self, 'E(%s) = %.15g  E_corr = %.15g',
                   self.__class__.__name__, self.e_tot, self.e_corr)
        return self.e_corr

def get_LT_data():

    tauarray=[-0.0089206000, -0.0611884000, -0.2313584000, -0.7165678000, -1.9685146000, -4.9561668000, -11.6625886000]
    weightarray=[0.0243048000, 0.0915096000, 0.2796534000, 0.7618910000, 1.8956444000, 4.3955808000, 9.6441228000]
    NLapPoints=len(tauarray)

    return (tauarray,weightarray,NLapPoints)

def get_batch(mem_avail,size,num_mat,override=0):

    mem_avail=float(mem_avail)
    max_batch_size=int(numpy.floor((mem_avail*(10.**6.)/8.)/(num_mat*size)))
    max_batch_size=numpy.amin(numpy.array([size,max_batch_size]))
    if override>0:
        max_batch_size=numpy.amin(numpy.array([size,override]))
    batch_num=int(numpy.ceil(float(size)/float(max_batch_size)))

    batch=(0,)
    ind=0
    while ind < size:
        batch+=(numpy.amin(numpy.array([ind+max_batch_size,size])),)
        ind+=max_batch_size

    return (max_batch_size,batch_num,batch)

def get_ao_batch(shell_data,max_mo_batch_size):

    #shell_data contains the shell indices (i.e., for s,p,s,p functions it will be [0,1,4,5,8])
    shell_data_save=shell_data.copy()
    shell_batch=(0,)
    ind=0
    while len(shell_data)>1:
        pos=numpy.where(shell_data<=max_mo_batch_size)[0][-1]
        ind+=pos
        shell_batch+=(ind,)
        shell_data-=shell_data[pos]
        shell_data=shell_data[pos:]
    ao_batch=tuple(numpy.array(shell_data_save)[[shell_batch]])

    return (shell_batch,ao_batch)

def compute_fft(func,coul,dim):

    dim_array=numpy.full(3,dim)
    small_dim_array=dim_array.copy()
    small_dim_array[2]=int(numpy.floor(dim_array[2]/2.0))+1
    
    func=pyfftw.interfaces.numpy_fft.rfftn(func.reshape(dim_array),dim_array,planner_effort='FFTW_MEASURE')
    func*=coul.reshape(small_dim_array)
    func=pyfftw.interfaces.numpy_fft.irfftn(func,dim_array,planner_effort='FFTW_MEASURE').flatten()

    return func

def kernel(mp, mo_energy=None, mo_coeff=None, verbose=logger.NOTE):

    nocc=mp.nocc
    nvir=mp.nmo-nocc
    if mo_energy is None: mo_energy=mp.mo_energy.reshape(-1,1)
    if mo_coeff is None: mo_coeff=mp.mo_coeff

    cell=mp._scf.cell
    mesh=cell.mesh
    dim=mesh[0]
    smalldim=int(numpy.floor(dim/2.0))+1
    ngs=dim*dim*dim

    coulG=pbctools.get_coulG(cell, mesh=mesh)
    coulG=coulG.reshape(mesh)
    coulG=coulG[:,:,:smalldim].reshape([dim*dim*smalldim])

    coords=cell.gen_uniform_grids(mesh=mesh)

    aoR=numint.eval_ao(cell, coords) #[ngs x nao]
    moR=numpy.asarray(pylib.dot(mo_coeff.T, aoR.T), order='C') #[nao x ngs]
    moRocc=moR[:nocc]
    moRvirt=moR[nocc:]

    (tauarray,weightarray,NLapPoints)=get_LT_data()

    #batching
    (max_grid_batch_size,grid_batch_num,grid_batch)=get_batch(mp.max_memory,ngs,5,override=0) #batching grid
    (max_mo_batch_size,mo_batch_num,mo_batch)=get_batch(mp.max_memory,nocc,2,override=20) #batching MO
    (shell_batch,ao_batch)=get_ao_batch(cell.ao_loc_nr(),max_mo_batch_size) #batching AO

    if grid_batch_num > 1:
        print "Splitting the grid into ", grid_batch_num, " batches of max size ", max_grid_batch_size
    if mo_batch_num > 1:
        print "Splitting the MOs into ", mo_batch_num, " batches of max size ", max_mo_batch_size

    Jtime=time.time()
    EMP2J=0.0
    for i in range(numpy.amin([NLapPoints,mp.lt_points])):
        Jint=0.0
        moRoccW=moRocc*numpy.exp(-mo_energy[:nocc]*tauarray[i]/2.) #[nocc x ngs]
        moRvirtW=moRvirt*numpy.exp(mo_energy[nocc:]*tauarray[i]/2.) #[nvirt x ngs]
        mo_occ=mo_coeff.T[:nocc]*numpy.exp(-mo_energy[:nocc]*tauarray[i]/2.)
        mo_virt=mo_coeff.T[nocc:]*numpy.exp(mo_energy[nocc:]*tauarray[i]/2.)
        for b1 in range(grid_batch_num):
            gbs=grid_batch[b1+1]-grid_batch[b1]

            g_o=numpy.zeros((gbs,ngs))
            for a1 in range(mo_batch_num):
                mbs=mo_batch[a1+1]-mo_batch[a1]
                moR_batch=numpy.zeros((mbs,ngs)) #[mbs x ngs]
                for a2 in range(len(shell_batch)-1):
                    aoR_batch=numint.eval_ao(cell, coords, shls_slice=(shell_batch[a2],shell_batch[a2+1])) #[ngs x shell_batch_size]
                    mo_occ_batch=mo_occ[mo_batch[a1]:mo_batch[a1+1],ao_batch[a2]:ao_batch[a2+1]]
                    moR_batch+=numpy.dot(mo_occ_batch,aoR_batch.T)
                g_o+=numpy.dot(moR_batch.T[grid_batch[b1]:grid_batch[b1+1]],moR_batch)

            #g_o=pylib.dot(moRoccW.T[grid_batch[b1]:grid_batch[b1+1]],moRoccW) #[gbs x ngs]
            g_v=pylib.dot(moRvirtW.T[grid_batch[b1]:grid_batch[b1+1]],moRvirtW) #[gbs x ngs]

            f=g_o*g_v #[gbs x ngs]
            g_o=g_v=None
            if mp.optimization=="Cython":
                F=numpy.zeros((gbs,ngs),dtype='float64') #[gbs x ngs]
                fft_cython.getJ(dim,gbs,f,F,coulG)
            elif mp.optimization=="Python":
                F=numpy.zeros((gbs,ngs),dtype='float64') #[gbs x ngs]
                for j in range(gbs):
                    F[j]=compute_fft(f[j,:],coulG,dim)
            else:
                raise RuntimeError('Only Cython and Python implemented!')
            f=None
            if grid_batch_num>1:
                F_T=numpy.zeros((ngs,gbs),dtype='float64')
                for b2 in range(grid_batch_num):
                    if b2!=b1:
                        gbs_in=grid_batch[b2+1]-grid_batch[b2]
                        g_o_in=pylib.dot(moRoccW.T[grid_batch[b2]:grid_batch[b2+1]],moRoccW) #[gbs_in x ngs]
                        g_v_in=pylib.dot(moRvirtW.T[grid_batch[b2]:grid_batch[b2+1]],moRvirtW) #[gbs_in x ngs]
                        f_in=g_o_in*g_v_in #[gbs x ngs]
                        g_o_in=g_v_in=None
                        if mp.optimization=="Cython":
                            F_in=numpy.zeros((gbs_in,ngs),dtype='float64') #[gbs_in x ngs]
                            fft_cython.getJ(dim,gbs_in,f_in,F_in,coulG)
                        elif mp.optimization=="Python":
                            F_in=numpy.zeros((gbs_in,ngs),dtype='float64') #[gbs_in x ngs]
                            for k in range(gbs_in):
                                F_in[k]=compute_fft(f_in[k,:],coulG,dim)
                        else:
                            raise RuntimeError('Only Cython and Python implemented!')
                        f_in=None
                        F_T[grid_batch[b2]:grid_batch[b2+1]]=F_in[:,grid_batch[b1]:grid_batch[b1+1]]
                        F_in=None
                    else:
                        F_T[grid_batch[b2]:grid_batch[b2+1]]=F[:,grid_batch[b1]:grid_batch[b1+1]]
                Jint+=numpy.sum(F_T.T*F)
                F=F_T=None
            else:
                Jint+=numpy.sum(F.T*F)
                F=F_T=None
        moRoccW=moRvirtW=None
        EMP2J-=2.*weightarray[i]*Jint*(cell.vol/ngs)**2.
        print EMP2J.real
    print "Took this long for J: ", time.time()-Jtime
    EMP2J=EMP2J.real
    print "EMP2J: ", EMP2J

    return EMP2J

cell=get_cell(10.26,'Si','diamond','gth-szv',4,'gth-pade',[2,2,2])
scf=get_scf(cell)
mp2=LTSOSMP2(scf)
mp2.optimization='Cython'
mp2.lt_points=1
mp2_energy=mp2.kernel()