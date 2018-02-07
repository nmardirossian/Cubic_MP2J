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

    #pragma omp parallel private(signalfft,resultfft,j,l)
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

double sumtrans_c(long int dim1, long int dim2, long int p1, long int p2, double* F1, double* F2){

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

    long int j, l;
    double *mat1;
    double *mat2;

    #pragma omp parallel private(mat1,mat2,j,l)
    #pragma omp for
    for (j=0; j<dim1; ++j){
        double *mat1=(double*) &F1[j*dim2];
        double *mat2=(double*) &F2[j*dim2];
        for (l=0; l<dim2; ++l){
            mat1[l]=mat1[l]*mat2[l];
        }
    }

    return;
}
