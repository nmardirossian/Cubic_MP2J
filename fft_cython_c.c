#include <omp.h>
#include <fftw3.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define REAL 0
#define IMAG 1
//#define TIMING 1

void getJ_c(long int batch, double* F, double* coulGsmall, long int* mesh, long int* smallmesh){

    long int j, l;
    double *signalfft;
    fftw_complex *resultfft;

#ifdef TIMING
    clock_t initsec, finalsec;
    double initfilltime=0.0;
    double ffttime=0.0;
    double coulGsmalltime=0.0;
    double iffttime=0.0;
    double finalfilltime=0.0;
#endif

    long int ngs=mesh[0]*mesh[1]*mesh[2];
    long int ngssmall=smallmesh[0]*smallmesh[1]*smallmesh[2];

    signalfft=(double*) fftw_malloc(sizeof(double)*ngs);
    resultfft=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*ngssmall);

    fftw_plan planfft = fftw_plan_dft_r2c_3d(mesh[0],mesh[1],mesh[2],
                                             signalfft,
                                             resultfft,
                                             FFTW_MEASURE);

    fftw_plan planifft = fftw_plan_dft_c2r_3d(mesh[0],mesh[1],mesh[2],
                                              resultfft,
                                              signalfft,
                                              FFTW_MEASURE);

    for (j=0; j<batch; ++j){

            // F[j]
#ifdef TIMING
            initsec=clock();
#endif
            for (l=0; l<ngs; ++l){
                signalfft[l]=F[j*ngs+l];
            }
#ifdef TIMING
            finalsec=clock();
            initfilltime+=(double)(finalsec-initsec)/CLOCKS_PER_SEC;
#endif

            // nm_fft(F[j])
#ifdef TIMING
            initsec=clock();
#endif
            fftw_execute(planfft);
#ifdef TIMING
            finalsec=clock();
            ffttime+=(double)(finalsec-initsec)/CLOCKS_PER_SEC;
#endif

            // nm_fft(F[:,j])*coulG
#ifdef TIMING
            initsec=clock();
#endif
            for (l=0; l<ngssmall; ++l){
                resultfft[l][REAL]=resultfft[l][REAL]*coulGsmall[l];
                resultfft[l][IMAG]=resultfft[l][IMAG]*coulGsmall[l];
            }
#ifdef TIMING
            finalsec=clock();
            coulGsmalltime+=(double)(finalsec-initsec)/CLOCKS_PER_SEC;
#endif

            // nm_ifft(nm_fft(F[:,j])*coulG)
#ifdef TIMING
            initsec=clock();
#endif
            fftw_execute(planifft);
#ifdef TIMING
            finalsec=clock();
            iffttime+=(double)(finalsec-initsec)/CLOCKS_PER_SEC;
#endif

            // F[j]
#ifdef TIMING
            initsec=clock();
#endif
            for (l=0; l<ngs; ++l){
                F[j*ngs+l]=signalfft[l]/(double)ngs;
            }
#ifdef TIMING
            finalsec=clock();
            finalfilltime+=(double)(finalsec-initsec)/CLOCKS_PER_SEC;
#endif

    }

#ifdef TIMING
    printf("Init fill took %f seconds\n",initfilltime);
    printf("FFT took %f seconds\n",ffttime);
    printf("coulGsmall took %f seconds\n",coulGsmalltime);
    printf("IFFT took %f seconds\n",iffttime);
    printf("Final fill took %f seconds\n",finalfilltime);
#endif

    fftw_destroy_plan(planfft);
    fftw_destroy_plan(planifft);
    fftw_free(signalfft);
    fftw_free(resultfft);

    return;

}
