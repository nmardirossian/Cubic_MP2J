#include <omp.h>
#include <fftw3.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define REAL 0
#define IMAG 1

void getJ_c(long int batch, fftw_complex* F, double* coulG, long int* mesh){

    long int j, l;
    long int ngs=mesh[0]*mesh[1]*mesh[2];

    fftw_complex *rawmem=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*ngs);
    fftw_complex *signalfft=(fftw_complex*) &rawmem[0];
    fftw_plan planfft = fftw_plan_dft_3d(mesh[0],mesh[1],mesh[2],signalfft,signalfft,FFTW_FORWARD,FFTW_MEASURE);
    fftw_plan planifft = fftw_plan_dft_3d(mesh[0],mesh[1],mesh[2],signalfft,signalfft,FFTW_BACKWARD,FFTW_MEASURE);

    #pragma omp parallel private(signalfft,l)
    #pragma omp for
    for (j=0; j<batch; ++j){
        fftw_complex *signalfft=(fftw_complex*) &F[j*ngs];
        fftw_execute_dft(planfft,signalfft,signalfft);
        for (l=0; l<ngs; ++l){
            signalfft[l][REAL]=signalfft[l][REAL]*coulG[l];
            signalfft[l][IMAG]=signalfft[l][IMAG]*coulG[l];
        }
        fftw_execute_dft(planifft,signalfft,signalfft);
        for (l=0; l<ngs; ++l){
            signalfft[l][REAL]=signalfft[l][REAL]/ngs;
            signalfft[l][IMAG]=0.0;
        }
    }

    fftw_destroy_plan(planfft);
    fftw_destroy_plan(planifft);
    fftw_free(rawmem);

    return;
}
