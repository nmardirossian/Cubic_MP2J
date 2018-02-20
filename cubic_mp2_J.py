import pyscf
import numpy
import h5py
import pyfftw
import time
import ase
import fft_cython
import sys
import os

from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools as pbctools
from pyscf import lib as pylib
from pyscf.pbc.dft import numint
from pyscf.lib import logger
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from pyscf.pbc.dft.gen_grid import gen_uniform_grids
pyfftw.interfaces.cache.enable()

OPTIMIZATION_TYPE="Cython"

def get_cell(lc_bohr,atom,unit_cell,basis,mesh,pseudo,supercell=None):

    cell=pbcgto.Cell()
    boxlen=lc_bohr
    ase_atom=ase.build.bulk(atom,unit_cell,a=boxlen)
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

    if supercell is not None:
        print "*****BUILDING SUPERCELL!*****"
        cell._built=False
        cell=pbctools.super_cell(cell,supercell)

    return cell

def get_scf(cell):

    scf=pbchf.RHF(cell,exxdiv=None)
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
        nocc=numpy.count_nonzero(mp.mo_occ>0)
        assert(nocc>0)
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
    def __init__(self,scf,mo_coeff=None,mo_occ=None):

        if mo_coeff is None: mo_coeff=scf.mo_coeff
        if mo_occ is None: mo_occ=scf.mo_occ

        self.mol=scf.mol
        self._scf=scf
        self.verbose=self.mol.verbose
        self.stdout=self.mol.stdout
        self.max_memory=scf.max_memory
        self.max_disk=scf.max_memory
        self.optimization='Cython'
        self.lt_points=7

##################################################
# don't modify the following attributes, they are not input options
        self.mo_energy=scf.mo_energy
        self.mo_coeff=mo_coeff
        self.mo_occ=mo_occ
        self._nocc=None
        self._nmo=None
        self._ngs=None
        self.e_corr=None
        self._keys=set(self.__dict__.keys())

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self,n):
        self._nocc=n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self,n):
        self._nmo=n

    @property
    def ngs(self):
        return self.get_ngs()
    @ngs.setter
    def ngs(self,n):
        self._ngs=n

    def dump_flags(self):
        log=logger.Logger(self.stdout,self.verbose)
        log.info('')
        log.info('******** %s flags ********',self.__class__)
        log.info('nocc=%s,nmo=%s,ngs=%s',self.nocc,self.nmo,self.ngs)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory,pylib.current_memory()[0])

    @property
    def emp2(self):
        return self.e_corr

    @property
    def e_tot(self):
        return self.e_corr + self._scf.e_tot

    get_nocc=get_nocc
    get_nmo=get_nmo
    get_ngs=get_ngs

    def kernel(self,mo_energy=None,mo_coeff=None):

        if mo_energy is not None: self.mo_energy=mo_energy
        if mo_coeff is not None: self.mo_coeff=mo_coeff
        if self.mo_energy is None or self.mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call scf.kernel() to generate them.')
        if self.verbose>=logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_corr=kernel(self,mo_energy,mo_coeff,self.verbose)
        logger.log(self,'E(%s)=%.15g  E_corr=%.15g',
                   self.__class__.__name__,self.e_tot,self.e_corr)
        return self.e_corr

def get_LT_data():

    #this should return the Laplace weights and roots
    #in the future, this should actually take as argument the orbital energies and fit a specified number of weights and roots

    tauarray=[-0.0089206000,-0.0611884000,-0.2313584000,-0.7165678000,-1.9685146000,-4.9561668000,-11.6625886000]
    weightarray=[0.0243048000,0.0915096000,0.2796534000,0.7618910000,1.8956444000,4.3955808000,9.6441228000]
    NLapPoints=len(tauarray)

    return (tauarray,weightarray,NLapPoints)

def get_batch(mem_avail,dim1,dim2,num_mat=1,num_mat_batch=1,override=0):

    #batches things based on available memory/disk, dimensions, number of times they need to be stored, etc.

    if override>0:
        max_batch_size=numpy.amin(numpy.array([dim1,override]))
        batch_num=int(numpy.ceil(float(dim1)/float(max_batch_size)))
    else:
        mem_avail=float(mem_avail)
        max_batch_size=int(numpy.floor((mem_avail*(10.**6.)/8.)/(num_mat*dim2)))
        max_batch_size=numpy.amin(numpy.array([dim1,max_batch_size]))
        batch_num=int(numpy.ceil(float(dim1)/float(max_batch_size)))
        if batch_num>1:
            max_batch_size=int(numpy.floor((mem_avail*(10.**6.)/8.)/(num_mat_batch*dim2)))
            max_batch_size=numpy.amin(numpy.array([dim1,max_batch_size]))
            batch_num=int(numpy.ceil(float(dim1)/float(max_batch_size)))
    batch=(0,)
    ind=0
    while ind<dim1:
        batch+=(numpy.amin(numpy.array([ind+max_batch_size,dim1])),)
        ind+=max_batch_size

    return (max_batch_size,batch_num,batch)

def get_ao_batch(shell_data,max_mo_batch_size):

    #specific batching for AOs

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

def get_disk_batch(grid_batch,ngs):

    #batching for the h5py columns

    disk_batch=list(grid_batch)
    while disk_batch[-1]<ngs:
        tmp=list(numpy.array(disk_batch)+disk_batch[-1])
        disk_batch=disk_batch+tmp[1:]
    disk_batch=numpy.array(disk_batch)
    disk_batch=disk_batch[disk_batch<ngs]
    disk_batch=tuple(list(disk_batch)+[ngs])
    disk_batch_num=len(disk_batch)-1

    return (disk_batch,disk_batch_num)

def python_fft(input,coulG,mesh,smallmesh):

    #for a row of f=g_o*g_v, compute F

    func=input.copy()
    func=pyfftw.interfaces.numpy_fft.rfftn(func.reshape(mesh),mesh,planner_effort='FFTW_MEASURE')
    func*=coulG.reshape(smallmesh)
    func=pyfftw.interfaces.numpy_fft.irfftn(func,mesh,planner_effort='FFTW_MEASURE').flatten()

    return func

def form_g(mp,mo,row,row_batch):

    #forms the occupied/virtual Green's function (either g_o or g_v)

    print "**********form_g**********"

    cell=mp._scf.cell
    coords=cell.gen_uniform_grids(mesh=cell.mesh) #[ngs x 3]
    nocc=mp.nocc
    nvirt=mp.nmo-nocc
    nmo=numpy.shape(mo)[1]

    (max_mo_batch_size,mo_batch_num,mo_batch)=get_batch(mp.max_memory,nmo,mp.ngs,num_mat=2,num_mat_batch=2,override=0) #batching MOs
    (max_ao_batch_size,_,_)=get_batch(mp.max_memory,mp.nmo,mp.ngs,num_mat=2,num_mat_batch=2,override=0) #batching AOs
    (shell_batch,ao_batch)=get_ao_batch(cell.ao_loc_nr(),max_ao_batch_size) #batching AOs

    rbs=row_batch[row+1]-row_batch[row]
    g=numpy.zeros((rbs,mp.ngs),dtype='float64') #[rbs x ngs]

    eval_ao_time=0.0 #TIMING
    pylib_dot_time=0.0 #TIMING
    for a1 in range(mo_batch_num):
        mbs=mo_batch[a1+1]-mo_batch[a1]
        moR_b=numpy.zeros((mp.ngs,mbs),dtype='float64') #[ngs x mbs]
        for a2 in range(len(shell_batch)-1):
            t=time.time() #TIMING
            aoR_b=numint.eval_ao(cell,coords,shls_slice=(shell_batch[a2],shell_batch[a2+1])) #[ngs x shell_batch_size]
            eval_ao_time+=time.time()-t #TIMING
            mo_b=mo[ao_batch[a2]:ao_batch[a2+1],mo_batch[a1]:mo_batch[a1+1]] #[shell_batch_size x mbs]
            t=time.time() #TIMING
            pylib.dot(aoR_b,mo_b,alpha=1,c=moR_b,beta=1) #[ngs x mbs]
            pylib_dot_time+=time.time()-t #TIMING
            aoR_b=mo_b=None
        t=time.time() #TIMING
        pylib.dot(moR_b[row_batch[row]:row_batch[row+1]],moR_b.T,alpha=1,c=g,beta=1) #[rbs x ngs]
        pylib_dot_time+=time.time()-t #TIMING
        moR_b=None

    coords=None

    if nmo==nocc:
        mo_str="occupied"
    elif nmo==nvirt:
        mo_str="virtual"

    if mo_batch_num>1:
        print "Splitting the "+mo_str+" MOs into ", mo_batch_num, " batches of max size ", max_mo_batch_size

    print "eval_ao_time took: ", eval_ao_time #TIMING
    print "pylib_dot took: ", pylib_dot_time #TIMING

    return g

def form_F(dim1,dim2,mat_out,mat_in):

    #forms the product of the occupied and virtual Green's functions, f=g_o*g_v
    #multiplies two matrices, mat_out and mat_in, storing the result in mat_out

    print "**********form_F**********"

    t=time.time() #TIMING
    if OPTIMIZATION_TYPE=="Cython":
        fft_cython.mult(dim1,dim2,mat_out,mat_in)
    elif OPTIMIZATION_TYPE=="Python":
        mat_out*=mat_in #[gbs x ngs]
    else:
        raise RuntimeError('Only Cython and Python implemented!')
    print "form_F took: ", time.time()-t #TIMING

    return mat_out

def fft_F(batch,F,coulG,mesh,smallmesh):

    #form the final F

    print "**********fft_F**********"

    t=time.time() #TIMING
    if OPTIMIZATION_TYPE=="Cython":
        fft_cython.getJ(batch,F,coulG,mesh,smallmesh)
    elif OPTIMIZATION_TYPE=="Python":
        for j in range(batch):
            F[j]=python_fft(F[j],coulG,mesh,smallmesh)
    else:
        raise RuntimeError('Only Cython and Python implemented!')
    print "fft_F took: ", time.time()-t #TIMING

    return F

def form_coulG(cell):

    #form coulG

    mesh=cell.mesh
    smallmesh=mesh.copy()
    smallmesh[-1]=int(numpy.floor(smallmesh[-1]/2.0))+1

    coulG=pbctools.get_coulG(cell,mesh=mesh) #[ngs]
    coulG=coulG.reshape(mesh) #[mesh[0] x mesh[1] x mesh[2]]
    coulG=coulG[:,:,:smallmesh[-1]].reshape([numpy.product(smallmesh),]) #[ngssmall]

    print "mesh: ", cell.mesh
    print "smallmesh: ", smallmesh

    return (coulG,mesh,smallmesh)

def init_h5py(row_batch,column_batch,F_h5py,F_T_h5py):

    #initialize the h5py files

    print "**********init_h5py**********"

    row_batch_num=len(row_batch)-1
    column_batch_num=len(column_batch)-1

    for b1 in range(row_batch_num):
        rbs=row_batch[b1+1]-row_batch[b1]
        for b2 in range(column_batch_num):
            cbs=column_batch[b2+1]-column_batch[b2]
            F_h5py.create_dataset("F_h5py_"+str(b1)+"_"+str(b2),(rbs,cbs),dtype='float64')
            F_T_h5py.create_dataset("F_T_h5py_"+str(b2)+"_"+str(b1),(cbs,rbs),dtype='float64')

    return None

def write_h5py(row,rbs,ngs,column_batch,F_h5py,F_T_h5py,F):

    #write to the initialized h5py files
    #this will only be triggered when the available disk space is larger than the available memory (and the latter is insufficient)
    #F (ngs x ngs) has been chopped up row-wise, and we are getting the 'row'th rbs x ngs piece
    #this function breaks the incoming rbs x ngs piece into smaller rbs x cbs pieces and saves them

    print "**********write_h5py**********"

    column_batch_num=len(column_batch)-1

    F_write_time=0.0 #TIMING
    F_T_time=0.0 #TIMING
    F_T_write_time=0.0 #TIMING
    for i in range(column_batch_num):
        cbs=column_batch[i+1]-column_batch[i]
        t=time.time() #TIMING
        F_h5py["F_h5py_"+str(row)+"_"+str(i)][:]=F[:,column_batch[i]:column_batch[i+1]]
        F_write_time+=time.time()-t #TIMING
        t=time.time() #TIMING
        if OPTIMIZATION_TYPE=="Cython":
            F_T=numpy.zeros((cbs,rbs),dtype='float64')
            fft_cython.trans(rbs,cbs,ngs,F_T,F[:,column_batch[i]:column_batch[i+1]])
        elif OPTIMIZATION_TYPE=="Python":
            F_T=F[:,column_batch[i]:column_batch[i+1]].T
        else:
            raise RuntimeError('Only Cython and Python implemented!')
        F_T_time+=time.time()-t #TIMING
        t=time.time() #TIMING
        F_T_h5py["F_T_h5py_"+str(i)+"_"+str(row)][:]=F_T
        F_T_write_time+=time.time()-t #TIMING
        F_T=None

    print "F_write took: ", F_write_time #TIMING
    print "F_T took: ", F_T_time #TIMING
    print "F_T_write took: ", F_T_write_time #TIMING

    return None

def read_h5py_sum(row_batch,column_batch,col_off,ts,scratch):

    print "**********read_h5py_sum**********"

    sum=0.0

    row_batch_num=len(row_batch)-1
    column_batch_num=len(column_batch)-1

    F_h5py=h5py.File(scratch+"/F_h5py_"+ts+".hdf5",'r')
    F_T_h5py=h5py.File(scratch+"/F_T_h5py_"+ts+".hdf5",'r')

    F1_read_time=0.0 #TIMING
    F2_read_time=0.0 #TIMING
    sum_time=0.0 #TIMING
    for b1 in range(row_batch_num):
        rbs=row_batch[b1+1]-row_batch[b1]
        for b2 in range(column_batch_num):
            cbs=column_batch[b2+1]-column_batch[b2]
            t=time.time() #TIMING
            F1=F_h5py["F_h5py_"+str(b1)+"_"+str(b2+col_off)][:]
            F1_read_time+=time.time()-t #TIMING
            t=time.time() #TIMING
            F2=F_T_h5py["F_T_h5py_"+str(b1+col_off)+"_"+str(b2)][:]
            F2_read_time+=time.time()-t #TIMING
            t=time.time() #TIMING
            if OPTIMIZATION_TYPE=="Cython":
                sum+=fft_cython.sum(rbs,cbs,cbs,cbs,F1,F2)
            elif OPTIMIZATION_TYPE=="Python":
                sum+=numpy.einsum('ij,ij->',F1,F2)
            else:
                raise RuntimeError('Only Cython and Python implemented!')
            sum_time+=time.time()-t #TIMING
            F1=F2=None
    F_h5py.close()
    F_T_h5py.close()

    print "F1_read took: ", F1_read_time #TIMING
    print "F2_read took: ", F2_read_time #TIMING
    print "sum took: ", sum_time #TIMING

    return sum

def read_h5py_T_sum(F1,row,rbs,ngs,column_batch,col_off,ts,scratch):

    print "**********read_h5py_T_sum**********"

    sum=0.0

    column_batch_num=len(column_batch)-1

    F_T_h5py=h5py.File(scratch+"/F_T_h5py_"+ts+".hdf5",'r')

    F2_read_time=0.0 #TIMING
    sum_time=0.0 #TIMING
    for b2 in range(column_batch_num):
        cbs=column_batch[b2+1]-column_batch[b2]
        t=time.time() #TIMING
        F2=F_T_h5py["F_T_h5py_"+str(row+col_off)+"_"+str(b2)][:]
        F2_read_time+=time.time()-t #TIMING
        t=time.time() #TIMING
        if OPTIMIZATION_TYPE=="Cython":
            sum+=fft_cython.sum(rbs,cbs,ngs,cbs,F1[:,column_batch[b2]:column_batch[b2+1]],F2)
        elif OPTIMIZATION_TYPE=="Python":
            sum+=numpy.einsum('ij,ij->',F1[:,column_batch[b2]:column_batch[b2+1]],F2)
        else:
            raise RuntimeError('Only Cython and Python implemented!')
        sum_time+=time.time()-t #TIMING
        F2=None
    F_T_h5py.close()

    print "F2_read took: ", F2_read_time #TIMING
    print "sum took: ", sum_time #TIMING

    return sum

def sum_trans(dim1,dim2,p1,p2,A,B):

    print "**********sum_trans**********"

    sum=0.0

    t=time.time() #TIMING
    if OPTIMIZATION_TYPE=="Cython":
        sum=fft_cython.sumtrans(dim1,dim2,p1,p2,A,B)
    elif OPTIMIZATION_TYPE=="Python":
        sum=numpy.einsum('ij,ji->',A,B)
    else:
        raise RuntimeError('Only Cython and Python implemented!')
    print "sum_trans took: ", time.time()-t #TIMING

    return sum

def kernel(mp,mo_energy=None,mo_coeff=None,verbose=logger.NOTE):

    #gaussian basis data
    nao=mp.nmo
    nocc=mp.nocc
    if mo_energy is None: mo_energy=mp.mo_energy #[nmo]
    if mo_coeff is None: mo_coeff=mp.mo_coeff #[nao x nmo]
    mo_energy=mo_energy.reshape(1,-1)

    #plane wave basis data
    ngs=mp.ngs
    cell=mp._scf.cell
    (coulG,mesh,smallmesh)=form_coulG(cell)

    #laplace transform data
    #TODO: allow this function to take number of laplace points as argument and actually fit the MO energies
    (tauarray,weightarray,NLapPoints)=get_LT_data()

    #batching grid
    #first, we check to see if we should do everything in memory.
    #it is done in memory if:
        #a). mp.max_memory is large enough to hold ngs x ngs x 2 doubles, regardless of the value of mp.max_disk
        #b). mp.max_memory>=mp.max_disk, which means to trigger the version that uses no disking, set mp.max_disk equal to mp.max_memory
    (max_grid_batch_size_disk,grid_batch_num_disk,grid_batch_disk)=get_batch(mp.max_memory,ngs,ngs,num_mat=2,num_mat_batch=3,override=0) #batching grid (mem)
    if (grid_batch_num_disk==1 or mp.max_memory>=mp.max_disk):
        print "[USING MEMORY ONLY]"
        storage="mem"
        if grid_batch_num_disk>1:
            print "Splitting the grid into ", grid_batch_num_disk, " outer batches of max size (mem) ", max_grid_batch_size_disk
    else:
        (max_grid_batch_size_disk,grid_batch_num_disk,grid_batch_disk)=get_batch(mp.max_disk,ngs,ngs,num_mat=2,num_mat_batch=2,override=0) #batching grid (disk)
        print "[USING DISK WITH MEMORY SUPPORT]"
        storage="disk"
        if grid_batch_num_disk>1:
            print "Splitting the grid into ", grid_batch_num_disk, " outer batches of max size (disk) ", max_grid_batch_size_disk

    #scratch directory and files
    try:
        scratchdir=os.environ['PYSCF_TMPDIR']
    except KeyError:
        scratchdir=''
        print "SCRATCH DIRECTORY NOT SET. WRITING TO CWD!"
    ts=time.strftime("%Y%m%d%H%M%S")

    E_MP2_J=0.0

    for i in range(numpy.amin([NLapPoints,mp.lt_points])):

        #J_LT is the correlation energy per Laplace point
        J_LT=0.0

        #weighted occupied and virtual MOs
        mo_occ=mo_coeff[:,:nocc]*numpy.exp(-mo_energy[:,:nocc]*tauarray[i]/2.) #[nao x nocc]
        mo_virt=mo_coeff[:,nocc:]*numpy.exp(mo_energy[:,nocc:]*tauarray[i]/2.) #[nao x nvirt]

        #outer loop over disk/mem
        for c1 in range(grid_batch_num_disk):

            print "********************OUTER DISK/MEM BATCH********************"

            #batching outer loop over mem
            gbsd=grid_batch_disk[c1+1]-grid_batch_disk[c1]
            if storage=="mem":
                (max_grid_batch_size,grid_batch_num,grid_batch)=get_batch(mp.max_memory,gbsd,ngs,num_mat=1,num_mat_batch=1,override=0) #batching grid (mem)
            elif storage=="disk":
                (max_grid_batch_size,grid_batch_num,grid_batch)=get_batch(mp.max_memory,gbsd,ngs,num_mat=2,num_mat_batch=2,override=0) #batching grid (mem)
            else:
                raise RuntimeError('Invalid option for storage!')
            grid_batch=tuple(numpy.array(grid_batch)+grid_batch_disk[c1])

            #initialize h5py files if disking
            if grid_batch_num>1:
                F_h5py=h5py.File(scratchdir+"/F_h5py_"+ts+".hdf5","w")
                F_T_h5py=h5py.File(scratchdir+"/F_T_h5py_"+ts+".hdf5","w")
                if c1==0:
                    (disk_batch,disk_batch_num)=get_disk_batch(grid_batch,ngs)
                    offset=grid_batch_num
                init_h5py(grid_batch,disk_batch,F_h5py,F_T_h5py)
                print "Splitting the grid into ", grid_batch_num, " inner batches of max size (mem) ", max_grid_batch_size

            #outer loop over mem
            for gb in range(grid_batch_num):

                print "********************OUTER MEM BATCH********************"

                gbs=grid_batch[gb+1]-grid_batch[gb]
                g_o=form_g(mp,mo_occ,gb,grid_batch)
                F=form_g(mp,mo_virt,gb,grid_batch)
                F=form_F(gbs,ngs,F,g_o)
                g_o=None
                F=fft_F(gbs,F,coulG,mesh,smallmesh)

                #write to h5py files if disking
                if grid_batch_num>1:
                    write_h5py(gb,gbs,ngs,disk_batch,F_h5py,F_T_h5py,F)
                    F=None

            #close h5py files if disking
            if grid_batch_num>1:
                F_h5py.close()
                F_T_h5py.close()

            if grid_batch_num_disk>1:

                #inner loop over disk/mem
                for c2 in range(grid_batch_num_disk):

                    print "********************INNER DISK/MEM BATCH********************"

                    if c2!=c1:

                        #batching inner loop over mem
                        gbsd_in=grid_batch_disk[c2+1]-grid_batch_disk[c2]
                        if storage=="mem":
                            (max_grid_batch_size_in,grid_batch_num_in,grid_batch_in)=get_batch(mp.max_memory,gbsd_in,ngs,num_mat=1,num_mat_batch=1,override=0) #batching grid (mem)
                        elif storage=="disk":
                            (max_grid_batch_size_in,grid_batch_num_in,grid_batch_in)=get_batch(mp.max_memory,gbsd_in,ngs,num_mat=2,num_mat_batch=2,override=0) #batching grid (mem)
                        else:
                            raise RuntimeError('Invalid option for storage!')
                        grid_batch_in=tuple(numpy.array(grid_batch_in)+grid_batch_disk[c2])

                        #inner loop over mem
                        for gb in range(grid_batch_num_in):

                            print "********************INNER MEM BATCH********************"

                            gbs_in=grid_batch_in[gb+1]-grid_batch_in[gb]
                            g_o=form_g(mp,mo_occ,gb,grid_batch_in)
                            F_in=form_g(mp,mo_virt,gb,grid_batch_in)
                            F_in=form_F(gbs_in,ngs,F_in,g_o)
                            g_o=None
                            F_in=fft_F(gbs_in,F_in,coulG,mesh,smallmesh)

                            #inner sum if c1!=c2
                            if grid_batch_num>1:
                                J_LT+=read_h5py_T_sum(F_in,gb,gbs_in,ngs,grid_batch,offset*c2,ts,scratchdir)
                            else:
                                J_LT+=sum_trans(gbs,gbs_in,ngs,ngs,F[:,grid_batch_in[gb]:grid_batch_in[gb+1]],F_in[:,grid_batch_disk[c1]:grid_batch_disk[c1+1]])
                            F_in=None

                    #inner sum if c1=c2
                    else:
                        if grid_batch_num>1:
                            J_LT+=read_h5py_sum(grid_batch,grid_batch,offset*c1,ts,scratchdir)
                        else:
                            J_LT+=sum_trans(gbs,gbs,ngs,ngs,F[:,grid_batch_disk[c1]:grid_batch_disk[c1+1]],F[:,grid_batch_disk[c1]:grid_batch_disk[c1+1]])

                F=None

            #outer sum
            else:
                if grid_batch_num>1:
                    J_LT+=read_h5py_sum(grid_batch,disk_batch,0,ts,scratchdir)
                else:
                    J_LT+=sum_trans(ngs,ngs,ngs,ngs,F,F)
                F=None

        E_MP2_J-=2.*weightarray[i]*J_LT*(cell.vol/ngs)**2.
        mo_occ=mo_virt=None
    E_MP2_J=E_MP2_J.real

    return E_MP2_J

mesh_val=int(sys.argv[1])
max_mem=int(sys.argv[2])
max_disk=int(sys.argv[3])
cell=get_cell(10.26,'Si','diamond','gth-szv',mesh_val,'gth-pade',supercell=[1,1,1])
scf=get_scf(cell)
mp2=LTSOSMP2(scf)
mp2.optimization='Cython'
mp2.lt_points=1
t1=time.time()
mp2.max_memory=max_mem
mp2.max_disk=max_disk
mp2_energy=mp2.kernel()
print "MP2 took: ", time.time()-t1
