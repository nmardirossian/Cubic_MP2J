#include <omp.h>
#include <fftw3.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define REAL 0
#define IMAG 1

void fftreal_c(long int batch, double* F, double* coulG, long int* mesh, long int* smallmesh){

    long int j, l;
    long int ngs=mesh[0]*mesh[1]*mesh[2];
    long int ngssmall=smallmesh[0]*smallmesh[1]*smallmesh[2];

    double *rawmem1=(double*) fftw_malloc(sizeof(double)*ngs);
    fftw_complex *rawmem2=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*batch*ngssmall);
    double *signalfft=(double*) &rawmem1[0];
    fftw_complex *resultfft=(fftw_complex*) &rawmem2[0];
    fftw_plan planfft = fftw_plan_dft_r2c_3d(mesh[0],mesh[1],mesh[2],signalfft,resultfft,FFTW_MEASURE);
    fftw_plan planifft = fftw_plan_dft_c2r_3d(mesh[0],mesh[1],mesh[2],resultfft,signalfft,FFTW_MEASURE);

    #pragma omp parallel private(signalfft,resultfft,j,l)
    #pragma omp for
    for (j=0; j<batch; ++j){
        double *signalfft=(double*) &F[j*ngs];
        fftw_complex *resultfft=(fftw_complex*) &rawmem2[j*ngssmall];
        fftw_execute_dft_r2c(planfft,signalfft,resultfft);
        for (l=0; l<ngssmall; ++l){
            resultfft[l][REAL]=resultfft[l][REAL]*coulG[l];
            resultfft[l][IMAG]=resultfft[l][IMAG]*coulG[l];
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

void fftcomp_c(long int batch, fftw_complex* F, double* coulG, long int* mesh){

    long int j, l;
    long int ngs=mesh[0]*mesh[1]*mesh[2];

    fftw_complex *rawmem=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*ngs);
    fftw_complex *signalfft=(fftw_complex*) &rawmem[0];
    fftw_plan planfft = fftw_plan_dft_3d(mesh[0],mesh[1],mesh[2],signalfft,signalfft,FFTW_FORWARD,FFTW_MEASURE);
    fftw_plan planifft = fftw_plan_dft_3d(mesh[0],mesh[1],mesh[2],signalfft,signalfft,FFTW_BACKWARD,FFTW_MEASURE);

    #pragma omp parallel private(signalfft,j,l)
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

double sum_c(long int dim1, long int dim2, long int p1, long int p2, double* F1, double* F2){

    //given two dim1 x dim2 matrices F1 and F2, returns sum of F1*F2
    //F1 and F2 are usually chunks from a dim1 x ngs matrix, hence the extra pointers

    long int j, l;
    double sum=0.0;

    #pragma omp parallel for reduction (+:sum)
    for (j=0; j<dim1; ++j){
        double insum=0.0;
        for (l=0; l<dim2; ++l){
            insum=insum+F1[j*p1+l]*F2[j*p2+l];
        }
        sum=sum+insum;
    }

    return sum;
}

double sumtrans_c(long int dim1, long int dim2, long int p1, long int p2, double* F1, double* F2){

    //given a dim1 x dim2 matrix F1, and a dim2 x dim1 matrix F2, returns sum of F1*F2.T
    //F1 and F2 are usually chunks from a dim1/dim2 x ngs matrix, hence the extra pointers

    long int j, l;
    double sum=0.0;

    #pragma omp parallel for reduction (+:sum)
    for (j=0; j<dim1; ++j){
        double insum=0.0;
        for (l=0; l<dim2; ++l){
            insum=insum+F1[j*p1+l]*F2[l*p2+j];
        }
        sum=sum+insum;
    }

    return sum;
}

void mult_c(long int dim1, long int dim2, double* F1, double* F2){

    //takes 2 dim1 x dim2 matrices (F1 and F2) and returns the Hadamard product in F1

    long int j, l;

    #pragma omp parallel private(j,l)
    #pragma omp for
    for (j=0; j<dim1; ++j){
        for (l=0; l<dim2; ++l){
            F1[j*dim2+l]=F1[j*dim2+l]*F2[j*dim2+l];
        }
    }

    return;
}

void trans_c(long int dim1, long int dim2, long int p, double* F1, double* F2){

    //takes a dim1 x dim2 matrix (F2) and stores its dim2 x dim1 transpose in F1
    //the matrix being transposed (F2) is a chunk of a larger dim1 x p matrix, hence the extra pointer

    long int j, l;

    #pragma omp parallel private(j,l)
    #pragma omp for
    for (j=0; j<dim1; ++j){
        for (l=0; l<dim2; ++l){
            F1[l*dim1+j]=F2[j*p+l];
        }
    }

    return;
}
