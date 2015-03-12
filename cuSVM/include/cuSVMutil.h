
#ifndef _CUSVMUTIL_H_
#define _CUSVMUTIL_H_


#define MBtoLeave         (200)

#define CUBIC_ROOT_MAX_OPS         (2000)

#define SAXPY_CTAS_MAX           (80)
#define SAXPY_THREAD_MIN         (32)
#define SAXPY_THREAD_MAX         (128)
#define TRANS_BLOCK_DIM             (16)

#ifdef _DEBUG

#define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;    \
    if( cudaSuccess != err) {                                                \
        printf( "Cuda errro in line " );              \
		char linebuffer [20];  sprintf(linebuffer,"%d" ,__LINE__ );									\
		printf( linebuffer  );              \
		printf( ": " );              \
		printf( cudaGetErrorString( err) );              \
		printf( ".\n" );              \
		getchar();                                                           \
                                              \
    } } while (0)

#define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        printf( "Cuda error in line " );              \
		char linebuffer [20];  sprintf(linebuffer,"%d" ,__LINE__ );									\
		printf( linebuffer  );              \
		printf( ": " );              \
		printf( cudaGetErrorString( err) );              \
		printf( ".\n" );              \
		getchar();                                                           \
                                                      \
    } } while (0)

#else  // not DEBUG

#  define CUDA_SAFE_CALL_NO_SYNC( call) call
#  define CUDA_SAFE_CALL( call) call

#endif



/*

void GPUtoMatlab(const float * gpupoint,const char *varname,int m, int n)
{

	mxArray *mexarray =mxCreateNumericMatrix(m, n,mxSINGLE_CLASS, mxREAL);
	float * mexpoint=(float *)mxGetData(mexarray);

	cudaMemcpy(mexpoint, gpupoint, sizeof(float)*m*n,cudaMemcpyDeviceToHost);

	 mexPutVariable("base",varname,mexarray);

}

void GPUtoMatlab(const int * gpupoint,const char *varname,int m, int n)
{

	mxArray *mexarray =mxCreateNumericMatrix(m, n,mxINT32_CLASS, mxREAL);
	int * mexpoint=(int *)mxGetData(mexarray);

	cudaMemcpy(mexpoint, gpupoint, sizeof(int)*m*n,cudaMemcpyDeviceToHost);

	 mexPutVariable("base",varname,mexarray);

}

void CPUtoMatlab(const float * cpupoint,const char *varname,int m, int n)
{

	mxArray *mexarray =mxCreateNumericMatrix(m, n,mxSINGLE_CLASS, mxREAL);
	float * mexpoint=(float *)mxGetData(mexarray);
	for (int k=0;k<m*n;k++)
	{
		mexpoint[k]=cpupoint[k];
	}

	 mexPutVariable("base",varname,mexarray);

}

void CPUtoMatlab(const int * cpupoint,const char *varname,int m, int n)
{

	mxArray *mexarray =mxCreateNumericMatrix(m, n,mxINT32_CLASS, mxREAL);
	int * mexpoint=(int *)mxGetData(mexarray);
	for (int k=0;k<m*n;k++)
	{
		mexpoint[k]=cpupoint[k];
	}

	 mexPutVariable("base",varname,mexarray);

}



void CPUtoMatlab(const unsigned int * cpupoint,const char *varname,int m, int n)
{

	mxArray *mexarray =mxCreateNumericMatrix(m, n,mxUINT32_CLASS, mxREAL);
	unsigned int * mexpoint=(unsigned int *)mxGetData(mexarray);
	for (int k=0;k<m*n;k++)
	{
		mexpoint[k]=cpupoint[k];
	}

	 mexPutVariable("base",varname,mexarray);

}*/


void checkCUDAError(const char *msg)
{
    cudaThreadSynchronize();
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {

        printf(msg);
        printf(" "); 
        printf(cudaGetErrorString( err) );
        printf(".  "); 
        exit(-1);
    }                         

}


void VectorSplay (int n, int tMin, int tMax, int gridW, int *nbrCtas, 
                        int *elemsPerCta, int *threadsPerCta)
{
    if (n < tMin) {
        *nbrCtas = 1;
        *elemsPerCta = n;
        *threadsPerCta = tMin;
    } else if (n < (gridW * tMin)) {
        *nbrCtas = ((n + tMin - 1) / tMin);
        *threadsPerCta = tMin;
        *elemsPerCta = *threadsPerCta;
    } else if (n < (gridW * tMax)) {
        int grp;
        *nbrCtas = gridW;
        grp = ((n + tMin - 1) / tMin);
        *threadsPerCta = (((grp + gridW -1) / gridW) * tMin);
        *elemsPerCta = *threadsPerCta;
    } else {
        int grp;
        *nbrCtas = gridW;
        *threadsPerCta = tMax;
        grp = ((n + tMin - 1) / tMin);
        grp = ((grp + gridW - 1) / gridW);
        *elemsPerCta = grp * tMin;
    }
}



__global__ void transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float block[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * TRANS_BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * TRANS_BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * TRANS_BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * TRANS_BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

#endif