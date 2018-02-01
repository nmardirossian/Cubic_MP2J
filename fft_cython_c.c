#include <omp.h>
#include <fftw3.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define REAL 0
#define IMAG 1

void getJ_c(long int batch, double* F, double* coulGsmall, long int* mesh, long int* smallmesh, long int* largemesh){

    long int j, l;
    long int ngs=mesh[0]*mesh[1]*mesh[2];
    long int ngssmall=smallmesh[0]*smallmesh[1]*smallmesh[2];
    long int ngslarge=largemesh[0]*largemesh[1]*largemesh[2];

    double *rawmem=(double*) fftw_malloc(sizeof(double)*ngslarge);
    double *signalfft=(double*) &rawmem[0];
    fftw_complex *resultfft=(fftw_complex*) &rawmem[0];
    fftw_plan planfft = fftw_plan_dft_r2c_3d(mesh[0],mesh[1],mesh[2],signalfft,resultfft,FFTW_MEASURE);
    fftw_plan planifft = fftw_plan_dft_c2r_3d(mesh[0],mesh[1],mesh[2],resultfft,signalfft,FFTW_MEASURE);

    #pragma omp parallel private(signalfft,resultfft,l)
    #pragma omp for
    for (j=0; j<batch; ++j){
        double *signalfft=(double*) &F[j*ngslarge];
        fftw_complex *resultfft=(fftw_complex*) &F[j*ngslarge];
        fftw_execute_dft_r2c(planfft,signalfft,resultfft);
        for (l=0; l<ngssmall; ++l){
            resultfft[l][REAL]=resultfft[l][REAL]*coulGsmall[l];
            resultfft[l][IMAG]=resultfft[l][IMAG]*coulGsmall[l];
        }
        fftw_execute_dft_c2r(planifft,resultfft,signalfft);
        for (l=0; l<ngslarge; ++l){
            signalfft[l]=signalfft[l]/ngs;
        }
    }

    fftw_destroy_plan(planfft);
    fftw_destroy_plan(planifft);
    fftw_free(rawmem);

    return;
}
