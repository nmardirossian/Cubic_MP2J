#include <omp.h>
#include <fftw3.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define REAL 0
#define IMAG 1

void getJ_c(long int batch, double* F, double* coulGsmall, long int* mesh, long int* smallmesh){

    long int j, l;
    long int ngs=mesh[0]*mesh[1]*mesh[2];
    long int ngssmall=smallmesh[0]*smallmesh[1]*smallmesh[2];

    double *rawmem1=(double*) fftw_malloc(sizeof(double)*ngs);
    fftw_complex *rawmem2=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*batch*ngssmall);
    double *signalfft=(double*) &rawmem1[0];
    fftw_complex *resultfft=(fftw_complex*) &rawmem2[0];
    fftw_plan planfft = fftw_plan_dft_r2c_3d(mesh[0],mesh[1],mesh[2],signalfft,resultfft,FFTW_MEASURE);
    fftw_plan planifft = fftw_plan_dft_c2r_3d(mesh[0],mesh[1],mesh[2],resultfft,signalfft,FFTW_MEASURE);

    #pragma omp parallel private(signalfft,resultfft,l)
    #pragma omp for
    for (j=0; j<batch; ++j){
        double *signalfft=(double*) &F[j*ngs];
        fftw_complex *resultfft=(fftw_complex*) &rawmem2[j*ngssmall];
        fftw_execute_dft_r2c(planfft,signalfft,resultfft);
        for (l=0; l<ngssmall; ++l){
            resultfft[l][REAL]=resultfft[l][REAL]*coulGsmall[l];
            resultfft[l][IMAG]=resultfft[l][IMAG]*coulGsmall[l];
        }
        fftw_execute_dft_c2r(planifft,resultfft,signalfft);
        for (l=0; l<ngs; ++l){
            signalfft[l]=signalfft[l]/ngs;
        }
    }

    fftw_destroy_plan(planfft);
    fftw_destroy_plan(planifft);
    fftw_free(rawmem1);
    fftw_free(rawmem2);

    return;
}
