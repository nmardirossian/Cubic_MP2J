#include <omp.h>
#include <fftw3.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define REAL 0
#define IMAG 1

void getJ_c(long int batch, double* F, double* coulGsmall, long int* mesh, long int* smallmesh, long int * largemesh){

    long int j, l;
    long int ngs=mesh[0]*mesh[1]*mesh[2];
    long int ngssmall=smallmesh[0]*smallmesh[1]*smallmesh[2];
    long int ngslarge=largemesh[0]*largemesh[1]*largemesh[2];

    int rank=3;
    int n[]={mesh[0],mesh[1],mesh[2]};
    long int howmany=batch;
    const int *inembed=NULL;
    const int *onembed=NULL;
    int istride=1;
    int ostride=1;
    long int idist=ngslarge;
    long int odist=ngslarge/2;

    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());

    double *rawmem=(double*) fftw_malloc(sizeof(double)*batch*ngslarge);

    double *signalfft=(double*) &rawmem[0];
    fftw_complex *resultfft=(fftw_complex*) &rawmem[0];

    fftw_plan planfft = fftw_plan_many_dft_r2c(rank,n,howmany,signalfft,inembed,istride,idist,resultfft,onembed,ostride,odist,FFTW_ESTIMATE);
    fftw_plan planifft = fftw_plan_many_dft_c2r(rank,n,howmany,resultfft,onembed,ostride,odist,signalfft,inembed,istride,idist,FFTW_ESTIMATE);

    for (j=0; j<batch; ++j){
        for (l=0; l<ngslarge; ++l){
            signalfft[j*ngslarge+l]=F[j*ngslarge+l];
        }
    }

    fftw_execute(planfft);

    for (j=0; j<batch; ++j){
        for (l=0; l<ngssmall; ++l){
            resultfft[j*ngssmall+l][REAL]=resultfft[j*ngssmall+l][REAL]*coulGsmall[l];
            resultfft[j*ngssmall+l][IMAG]=resultfft[j*ngssmall+l][IMAG]*coulGsmall[l];
        }
    }

    fftw_execute(planifft);

    for (j=0; j<batch; ++j){
        for (l=0; l<ngslarge; ++l){
            F[j*ngslarge+l]=signalfft[j*ngslarge+l]/(double)ngs;
        }
    }

    fftw_destroy_plan(planfft);
    fftw_destroy_plan(planifft);
    fftw_free(rawmem);
    fftw_cleanup_threads();

    return;
}
