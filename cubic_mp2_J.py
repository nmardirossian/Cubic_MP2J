import pyscf
import numpy
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

def get_MP2J(moRocc,moRvirt,coulGsmall):

	tauarray=[-0.0089206000, -0.0611884000, -0.2313584000, -0.7165678000, -1.9685146000, -4.9561668000, -11.6625886000]
	weightarray=[0.0243048000, 0.0915096000, 0.2796534000, 0.7618910000, 1.8956444000, 4.3955808000, 9.6441228000]
	NLapPoints=len(tauarray)

	if (numpy.shape(moRocc)[1]==numpy.shape(moRvirt)[1]):
		ngs=numpy.shape(moRocc)[1]
		dim=int(numpy.cbrt(ngs))

	print (ngs,dim)

	Jtime=time.time()
	EMP2J=0.0
#	for i in range(NLapPoints):
	for i in range(1):
		print EMP2J.real

		moRoccW=moRocc*numpy.exp(-orben[:nocc]*tauarray[i]/2.)
		moRvirtW=moRvirt*numpy.exp(orben[nocc:]*tauarray[i]/2.)

		gfunc=pylib.dot(moRoccW.T,moRoccW)
		gbarfunc=pylib.dot(moRvirtW.T,moRvirtW)
		ffunc=gfunc*gbarfunc

		gfunc=gbarfunc=None
		moRoccW=moRvirtW=None

		f2func=numpy.zeros((ngs,ngs),dtype='float64')

		print "Entering cython J"
		timeit=time.time()
		fft_cython.getJ(dim,ngs,ffunc,f2func,coulGsmall)
		print "Cython J call took: ", time.time()-timeit

		ffunc=None
		f2func=f2func*f2func.T
		Jint=numpy.sum(f2func)
		EMP2J-=2.*weightarray[i]*Jint*(cell.vol/ngs)**2.

		f2func=None

	print "Took this long for J: ", time.time()-Jtime
	EMP2J=EMP2J.real
	print "EMP2J: ", EMP2J

	return EMP2J



cell=get_cell(6.74,'C','diamond','gth-szv',3,'gth-pade')
(mo,orben,nocc)=get_orbitals(cell)
(moRocc,moRvirt,coulGsmall)=get_moR(cell,mo,nocc)
EMP2J=get_MP2J(moRocc,moRvirt,coulGsmall)