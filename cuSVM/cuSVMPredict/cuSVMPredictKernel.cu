

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
//#include <cutil.h> removed in CUDA 5.0
#include <math.h>
#include "cublas.h"  
#include "cuda.h" 
#include "../include/cuSVMutil.h"


/*This mixed-precision matrix-vector multiplication algorithm is based on cublasSgemv NVIDIA's CUBLAS 1.1.
In his tests, the author has found catastrophic prediction errors resulting from using only single precision floating point arithmetic
for the multiplication of the predictive kernel matrix by the SVM coefficients; however, all of the errors he found disappeared when he switched to 
a mixed-precision approach where the scalar dot-product accumulator is a double precision number.  Thus, the use of full double precision 
arithmetic, which would involve significant performance penalties, does not seem necessary.
CUBLAS 1.1 source code is available at: http://forums.nvidia.com/index.php?showtopic=59101, and
CUBLAS is available at http://www.nvidia.com/cuda .*/


#define LOG_THREAD_COUNT    (7)
#define THREAD_COUNT        (1 << LOG_THREAD_COUNT)
#define CTAS                (64) 
#define IDXA(row,col)       (lda*(col)+(row))
#define IDXX(i)             (startx + ((i) * incx))
#define IDXY(i)             (starty + ((i) * incy))
#define TILEW_LOG           (5)
#define TILEW               (1 << TILEW_LOG)
#define TILEH_LOG           (5)
#define TILEH               (1 << TILEH_LOG)
#define X_ELEMS_PER_THREAD  (4)
#define IINC                (CTAS * THREAD_COUNT)
#define JINC                (THREAD_COUNT * X_ELEMS_PER_THREAD)
#define XINC                (THREAD_COUNT)


__shared__ float XX[TILEH];             
__shared__ float AA[(TILEH+1)*TILEW];   


__global__ void sgemvn_mixedprecis(const float *A, const float *x,float *y, int m, int n, int lda, int   incx,   int   incy) 
{
    __shared__ float XX[JINC];
    int i, ii, j, jj, idx, incr, tid;
    double sdot;
    int startx;
    int starty;


    tid = threadIdx.x;
    startx = (incx >= 0) ? 0 : ((1 - n) * incx);
    starty = (incy >= 0) ? 0 : ((1 - m) * incy);

    for (i = 0; i < m; i += IINC) {
 
        ii = i + blockIdx.x * THREAD_COUNT;            
        if (ii >= m) break; 
        ii += tid; 
        sdot = 0.0f; 

        for (j = 0; j < n; j += JINC) {
            int jjLimit = min (j + JINC, n);
            incr = XINC * incx;
            jj = j + tid;
            __syncthreads ();
            idx = IDXX(jj);

            if (jj < (jjLimit - 3 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
                XX[tid+1*XINC] = x[idx + 1 * incr];
                XX[tid+2*XINC] = x[idx + 2 * incr];
                XX[tid+3*XINC] = x[idx + 3 * incr];
            }
            else if (jj < (jjLimit - 2 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
                XX[tid+1*XINC] = x[idx + 1 * incr];
                XX[tid+2*XINC] = x[idx + 2 * incr];
            }
            else if (jj < (jjLimit - 1 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
                XX[tid+1*XINC] = x[idx + 1 * incr];
            }
            else if (jj < (jjLimit - 0 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
            }

            __syncthreads ();
            
            if (ii < m) { /* if this row is active, accumulate dp */
                idx = IDXA(ii, j);
                incr = lda;
                jjLimit = jjLimit - j;
                jj = 0;
                while (jj < (jjLimit - 5)) {
                    sdot += A[idx + 0*incr] * XX[jj+ 0];
                    sdot += A[idx + 1*incr] * XX[jj+ 1];
                    sdot += A[idx + 2*incr] * XX[jj+ 2];
                    sdot += A[idx + 3*incr] * XX[jj+ 3];
                    sdot += A[idx + 4*incr] * XX[jj+ 4];
                    sdot += A[idx + 5*incr] * XX[jj+ 5];
                    jj   += 6;
                    idx  += 6 * incr;
                }
                while (jj < jjLimit) {
                    sdot += A[idx + 0*incr] * XX[jj+ 0];
                    jj   += 1;
                    idx  += 1 * incr;
                }
            }
        }
        if (ii < m) {
            idx = IDXY(ii);
     
           y[idx] = sdot;
        }
    }
}



//The memory access pattern and structure of this code is derived from Vasily Volkov's highly
// optimized matrix-matrix multiply CUDA code.
//His website is http://www.cs.berkeley.edu/~volkov/

__global__ void RBFKernelForPredict( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float kernelwidth )
{	

    int inx = threadIdx.x;
	int iny = threadIdx.y;
	int ibx = blockIdx.x * 32;
	int iby = blockIdx.y * 32;
	
	A += ibx + inx + __mul24( iny, lda );
	B += iby + inx + __mul24( iny, ldb );
	C += ibx + inx + __mul24( iby + iny, ldc );
	
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	for( int i = 0; i < k; i += 4 )
	{
		__shared__ float a[4][32];
		__shared__ float b[4][32];
		
		a[iny][inx] = A[i*lda];
		a[iny+2][inx] = A[(i+2)*lda];
		b[iny][inx] = B[i*ldb];
		b[iny+2][inx] = B[(i+2)*ldb];
		__syncthreads();
		
		for( int j = 0; j < 4; j++ )
		{
			float _a = a[j][inx];
			float *_b = &b[j][0] + iny;
            float _asquared=_a*_a;;    	
            
            
            //The (negative here) squared distance between datapoints is necessary for the calculation of the RBF Kernel.
            //This code uses the identity -(x-y)^2=2*x*y-x^2-y^2.   		

			c[0] += 2.f*_a*_b[0]-_asquared-_b[0]*_b[0];
			c[1] += 2.f*_a*_b[2]-_asquared-_b[2]*_b[2];
			c[2] += 2.f*_a*_b[4]-_asquared-_b[4]*_b[4];
			c[3] += 2.f*_a*_b[6]-_asquared-_b[6]*_b[6];
			c[4] += 2.f*_a*_b[8]-_asquared-_b[8]*_b[8];
			c[5] += 2.f*_a*_b[10]-_asquared-_b[10]*_b[10];
			c[6] += 2.f*_a*_b[12]-_asquared-_b[12]*_b[12];
			c[7] += 2.f*_a*_b[14]-_asquared-_b[14]*_b[14];
			c[8] += 2.f*_a*_b[16]-_asquared-_b[16]*_b[16];
			c[9] += 2.f*_a*_b[18]-_asquared-_b[18]*_b[18];
			c[10] += 2.f*_a*_b[20]-_asquared-_b[20]*_b[20];
			c[11] += 2.f*_a*_b[22]-_asquared-_b[22]*_b[22];
			c[12] += 2.f*_a*_b[24]-_asquared-_b[24]*_b[24];
			c[13] += 2.f*_a*_b[26]-_asquared-_b[26]*_b[26];
			c[14] += 2.f*_a*_b[28]-_asquared-_b[28]*_b[28];
			c[15] += 2.f*_a*_b[30]-_asquared-_b[30]*_b[30];
		}
		__syncthreads();
	}
	

	for( int i = 0; i < 16; i++, C += 2*ldc )
        // Here the negative squared distances between datapoints, calculated above, are multiplied by the kernel width parameter and exponentiated.
		C[0] = exp(kernelwidth*c[i]); 
  
}



extern "C"
void GPUPredictWrapper(int m, int n, int k, float kernelwidth, const float *Test, const float *Svs, float * alphas,float *prediction, float beta,float isregression)
{	
    

    mxArray *mexelapsed = mxCreateNumericMatrix(1, 1,mxSINGLE_CLASS, mxREAL);
	float * elapsed=(float *)mxGetData(mexelapsed);    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
        
    int paddedm=m+32-m%32;
    int paddedk=k+32-k%32;
    int paddedn=n+32-n%32;    
      
 


    float* d_PaddedSvs;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_PaddedSvs, paddedn*paddedk*sizeof(float)));

    CUDA_SAFE_CALL( cudaMemset(d_PaddedSvs,0.f,paddedn*paddedk*sizeof(float)));
    CUDA_SAFE_CALL( cudaMemcpy(d_PaddedSvs, Svs, sizeof(float)*n*k,cudaMemcpyHostToDevice));
  
    float* d_PaddedSvsT;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_PaddedSvsT, paddedn*paddedk*sizeof(float)));
  
    CUDA_SAFE_CALL(  cudaMemset(d_PaddedSvsT,0.f,paddedn*paddedk*sizeof(float)));
    dim3 gridtranspose(ceil((double)n / TRANS_BLOCK_DIM), ceil((double)paddedk / TRANS_BLOCK_DIM), 1);
    dim3 threadstranspose(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM, 1);    
    transpose<<< gridtranspose, threadstranspose >>>(d_PaddedSvsT, d_PaddedSvs, n,paddedk);
    dim3 gridtranspose2(ceil((double)paddedk / TRANS_BLOCK_DIM), ceil((double)paddedn / TRANS_BLOCK_DIM), 1);
    transpose<<< gridtranspose2, threadstranspose >>>(d_PaddedSvs, d_PaddedSvsT, paddedk,paddedn);
    CUDA_SAFE_CALL(  cudaFree(d_PaddedSvsT));


 
    double DoubleNecIterations=(double)paddedm/CUBIC_ROOT_MAX_OPS;
    DoubleNecIterations*=(double)paddedn/CUBIC_ROOT_MAX_OPS;
    DoubleNecIterations*=(double)paddedk/CUBIC_ROOT_MAX_OPS;        
    int NecIterations=ceil(DoubleNecIterations);
    int RowsPerIter=ceil((double)paddedm/NecIterations)+32-int(ceil((double)paddedm/NecIterations))%32;
    NecIterations=ceil((double)paddedm/RowsPerIter);


   
	dim3 grid( RowsPerIter/32, paddedn/32, 1 );
	dim3 threads2( 32, 2, 1 );


    float * d_TestInter;
    float * d_QInter;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_TestInter, RowsPerIter*paddedk*sizeof(float)));
   
    CUDA_SAFE_CALL(cudaMemset(d_TestInter,0.f,RowsPerIter*paddedk*sizeof(float)));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_QInter, RowsPerIter*paddedn*sizeof(float)));
    
    
    float * d_alphas;
    float * d_prediction;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_alphas, n*sizeof(float)));
    cublasSetVector(n,sizeof(float),alphas,1,d_alphas,1);
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_prediction, NecIterations*RowsPerIter*sizeof(float)));
        

    for (int j=0;j<NecIterations;j++)
    {
       
        if (j+1==NecIterations)
        {
           cublasSetMatrix(m-j*RowsPerIter,k,sizeof(float),Test+j*RowsPerIter,m,d_TestInter,RowsPerIter);
          
        }
        else
        {
            cublasSetMatrix(RowsPerIter,k,sizeof(float),Test+j*RowsPerIter,m,d_TestInter,RowsPerIter);
        }
             
        RBFKernelForPredict<<<grid, threads2>>>(d_TestInter, RowsPerIter, d_PaddedSvs, paddedn, d_QInter, RowsPerIter, paddedk, kernelwidth);
            

         sgemvn_mixedprecis<<<CTAS,THREAD_COUNT>>>(d_QInter,d_alphas,d_prediction+j*RowsPerIter,RowsPerIter,n,RowsPerIter,1,1);
        
    }


    cublasGetVector(m,sizeof(float),d_prediction,1,prediction,1);
    


    for(int j=0;j<m;j++)
    {
        prediction[j]+=beta;
        if (isregression!=1){prediction[j]=prediction[j]<0?-1.0:1.0;}    
    }

    

   

    cudaThreadSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed, start, stop);
   
     mexPutVariable("base","cuSVMPredictTimeInMS",mexelapsed);
    

   
    CUDA_SAFE_CALL( cudaFree(d_alphas));
    CUDA_SAFE_CALL( cudaFree(d_TestInter));
    CUDA_SAFE_CALL( cudaFree(d_QInter));
    CUDA_SAFE_CALL( cudaFree(d_PaddedSvs));
    CUDA_SAFE_CALL(cudaFree(d_prediction));

    CUDA_SAFE_CALL(cudaThreadExit());
}	

