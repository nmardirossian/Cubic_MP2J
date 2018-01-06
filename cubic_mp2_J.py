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
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

def get_cell(lc_bohr,atom,unit_cell,basis,gs,pseudo):

	#TODO Make it work with supercell
	cell=pbcgto.Cell()
	boxlen=lc_bohr
	ase_atom=ase.build.bulk(atom, unit_cell, a=boxlen)
	cell.a=ase_atom.cell
	cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
	cell.basis=basis
	cell.charge=0
	cell.dimension=3
	cell.incore_anyway=False
	cell.gs=numpy.array([gs,gs,gs])
	cell.max_memory=8000
	cell.pseudo=pseudo
	cell.spin=0
	cell.unit='B'
	cell.verbose=10
	cell.build()

	return cell

def get_orbitals(cell):

	mf=pbchf.RHF(cell, exxdiv=None)
	mf.conv_tol=1e-10
	mf.conv_tol_grad=1e-8
	mf.diis=True
	mf.diis_space=20
	mf.direct_scf_tol=1e-14
	mf.init_guess='minao'
	mf.max_cycle=200
	mf.max_memory=8000
	mf.verbose=10
	mf.scf()

	mo=mf.mo_coeff
	orben=mf.mo_energy
	orben=orben.reshape(-1,1)
	nelec=cell.nelectron
	nocc=nelec/2

	return (mo,orben,nocc)

def get_moR(cell,mo,nocc):
	with_df=pbcdf.FFTDF(cell)
	coords=cell.gen_uniform_grids(with_df.gs)
	gs=with_df.gs[0]
	dim=2*gs+1
	ngs=dim*dim*dim
	smalldim=int(numpy.floor(dim/2.0))+1

	aoR=with_df._numint.eval_ao(cell, coords)[0]
	moR=numpy.asarray(pylib.dot(mo.T, aoR.T), order='C')
	moRocc=moR[:nocc]
	moRvirt=moR[nocc:]

	coulG=pbctools.get_coulG(cell, gs=with_df.gs)
	coulGsmall=coulG.reshape([dim,dim,dim])
	coulGsmall=coulGsmall[:,:,:smalldim].reshape([dim*dim*smalldim])

	return (moRocc,moRvirt,coulGsmall)

def get_LT_data():

	tauarray=[-0.0089206000, -0.0611884000, -0.2313584000, -0.7165678000, -1.9685146000, -4.9561668000, -11.6625886000]
	weightarray=[0.0243048000, 0.0915096000, 0.2796534000, 0.7618910000, 1.8956444000, 4.3955808000, 9.6441228000]
	NLapPoints=len(tauarray)

	return (tauarray,weightarray,NLapPoints)

def get_MP2J(moRocc,moRvirt,coulGsmall):

	(tauarray,weightarray,NLapPoints)=get_LT_data()

	if (numpy.shape(moRocc)[1]==numpy.shape(moRvirt)[1]):
		ngs=numpy.shape(moRocc)[1]
		dim=int(numpy.cbrt(ngs))

	Jtime=time.time()
	EMP2J=0.0
	for i in range(NLapPoints):

		moRoccW=moRocc*numpy.exp(-orben[:nocc]*tauarray[i]/2.)
		moRvirtW=moRvirt*numpy.exp(orben[nocc:]*tauarray[i]/2.)

		gfunc=pylib.dot(moRoccW.T,moRoccW)
		gbarfunc=pylib.dot(moRvirtW.T,moRvirtW)
		ffunc=gfunc*gbarfunc

		gfunc=gbarfunc=None
		moRoccW=moRvirtW=None

		f2func=numpy.zeros((ngs,ngs),dtype='float64')

		timeit=time.time()
		fft_cython.getJ(dim,ngs,ffunc,f2func,coulGsmall)
		print "Cython J call took: ", time.time()-timeit

		ffunc=None
		f2func=f2func*f2func.T
		Jint=numpy.sum(f2func)
		EMP2J-=2.*weightarray[i]*Jint*(cell.vol/ngs)**2.

		f2func=None

		print EMP2J.real

	print "Took this long for J: ", time.time()-Jtime
	EMP2J=EMP2J.real
	print "EMP2J: ", EMP2J

	return EMP2J

pyfftw.interfaces.cache.enable()
def nm_fft(inp,ngs):
	griddim=numpy.full(3,int(numpy.cbrt(ngs)))

	return pyfftw.interfaces.numpy_fft.rfftn(inp.reshape(griddim),griddim,planner_effort='FFTW_MEASURE').flatten()

def nm_ifft(inp,ngs):
	griddim=numpy.full(3,int(numpy.cbrt(ngs)))

	return pyfftw.interfaces.numpy_fft.irfftn(inp.reshape(griddim),griddim,planner_effort='FFTW_MEASURE').flatten()

def get_batch_info(mem_avail,ngs):

	#input is available memory in GB

	bsize=int(numpy.floor(numpy.amin(numpy.array([ngs,mem_avail*(10**9)*2/(8*ngs)]))))
	bnum=int(numpy.ceil(ngs/float(bsize)))

	return (bsize,bnum)

def get_dim_from_ngs(ngs):

	dim=numpy.full(3,int(numpy.cbrt(ngs)))

	return dim

def get_small_dim_from_ngs(ngs):

	dim=get_dim_from_ngs(ngs)
	dim[2]=int(numpy.floor(dim[2]/2.0))+1

	return dim

def compute_fft(func,coul):
	dim=get_dim_from_ngs(len(func))
	smalldim=get_small_dim_from_ngs(len(func))
	
	tmp=pyfftw.interfaces.numpy_fft.rfftn(func.reshape(dim),dim,planner_effort='FFTW_MEASURE')
	tmp*=coul.reshape(smalldim)
	tmp=pyfftw.interfaces.numpy_fft.irfftn(tmp,dim,planner_effort='FFTW_MEASURE').flatten()

	return tmp
	

def get_MP2J_linmem(moRocc,moRvirt,coulGsmall,mem_avail):

	(tauarray,weightarray,NLapPoints)=get_LT_data()

	if (numpy.shape(moRocc)[1]==numpy.shape(moRvirt)[1]):
		ngs=numpy.shape(moRocc)[1]
		dim=int(numpy.cbrt(ngs))

	(bsize,bnum)=get_batch_info(mem_avail,ngs)
	print "Splitting the task into ", bnum, " batches of size ", bsize

	Jtime=time.time()
	EMP2J=0.0 #accumulate energy over all Laplace points
	for i in range(NLapPoints):
		Jint=0.0 #acculumlate energy over a single Laplace point
		moRoccW=moRocc*numpy.exp(-orben[:nocc]*tauarray[i]/2.) #phi_occ*exp(-eps*tau/2); [nocc x ngs]
		moRvirtW=moRvirt*numpy.exp(orben[nocc:]*tauarray[i]/2.) #phi_virt*exp(eps*tau/2); [nvirt x ngs]
		for b1 in range(bnum):
			g_o=pylib.dot(moRoccW.T[b1*bsize:(b1+1)*bsize],moRoccW) #[cur_bsize x ngs]
			g_v=pylib.dot(moRvirtW.T[b1*bsize:(b1+1)*bsize],moRvirtW) #[cur_bsize x ngs]
			f=g_o*g_v #[cur_bsize x ngs]
			cur_bsize=numpy.shape(f)[0] #size of the current batch (always bsize except maybe on last batch)
			F=numpy.zeros((cur_bsize,ngs),dtype='complex128') #[cur_bsize x ngs]
			for j in range(cur_bsize):
				F[j]=compute_fft(f[j,:],coulGsmall)
			if bnum>1:
				F_T=numpy.zeros((ngs,cur_bsize),dtype='complex128')
				for b2 in range(bnum):
					if b2!=b1:
						g_o_in=pylib.dot(moRoccW.T[b2*bsize:(b2+1)*bsize],moRoccW)
						g_v_in=pylib.dot(moRvirtW.T[b2*bsize:(b2+1)*bsize],moRvirtW)
						f_in=g_o_in*g_v_in
						cur_bsize_in=numpy.shape(f_in)[0]
						F_in=numpy.zeros((cur_bsize_in,ngs),dtype='complex128')
						for j in range(cur_bsize_in):
							F_in[j]=compute_fft(f_in[j,:],coulGsmall)
						F_T[b2*bsize:(b2+1)*bsize]=F_in[:,b1*bsize:(b1+1)*bsize]
					else:
						F_T[b2*bsize:(b2+1)*bsize]=F[:,b1*bsize:(b1+1)*bsize]
				Jint+=numpy.sum(F_T.T*F)
			else:
				Jint+=numpy.sum(F.T*F)
		EMP2J-=2*weightarray[i]*Jint*(cell.vol/ngs)**2
		print EMP2J.real
	print "Took this long for J: ", time.time()-Jtime

	return EMP2J.real

cell=get_cell(6.74,'C','diamond','gth-szv',6,'gth-pade')
(mo,orben,nocc)=get_orbitals(cell)
(moRocc,moRvirt,coulGsmall)=get_moR(cell,mo,nocc)
EMP2J=get_MP2J(moRocc,moRvirt,coulGsmall)
EMP2J_linmem=get_MP2J_linmem(moRocc,moRvirt,coulGsmall,4.0/500.0)

if (EMP2J==EMP2J_linmem):
	print "O(N^2) and O(N) mem versions agree!"
else:
	print "Difference is: ", EMP2J-EMP2J_linmem