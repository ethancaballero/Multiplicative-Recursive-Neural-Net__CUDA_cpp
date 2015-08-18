#include "kernels.h"

#include <ctime>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

#include "helper_cuda.h"

/***** CUDA Kernels *****/

__global__ void
setupRandomVectorGen(curandState* state, unsigned long seed, unsigned int numElems) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems) return;
    curand_init(seed, id, 0, &(state[id]));
}

__global__ void
runRandomVectorGen(float* vec, curandState* globalState, float threshold, unsigned int numElems) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems) return;
    curandState localState = globalState[id];
    float rndVal = curand_uniform(&localState);
    
    vec[id] = (rndVal * 2 * threshold) - threshold;
}

__global__ void
updateParams(float* params, float* derivatives, float* weights, float learnRate, unsigned int numElems) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems) return;
    float epsilon = 0.0001;
    weights[id] += derivatives[id]*derivatives[id];
    params[id] -= (learnRate * derivatives[id])/(sqrt(weights[id]) + epsilon);
}

/***** Kernel Wrappers *****/

void
kernelRandomwordVecs(ParamMem_t& params, float threshold) {
    timeval tim;
    gettimeofday(&tim, NULL);
    double t1=tim.tv_sec+(tim.tv_usec/1000000.0);

    unsigned int blockSize = 1024;
    unsigned int numElems = params.nWords * params.wordVecDim;
    unsigned int numBlocks = numElems / blockSize + 1;
    dim3 threadsPerBlock(blockSize, 1, 1);
    curandState* devState;
    checkCudaErrors(cudaMalloc((void**)&devState, numElems*sizeof(curandState)));
    setupRandomVectorGen<<<numBlocks, threadsPerBlock>>>(devState, time(NULL), numElems);
    runRandomVectorGen<<<numBlocks, threadsPerBlock>>>(params.wordVecs, devState, threshold, numElems);
    
    checkCudaErrors(cudaFree(devState));

    gettimeofday(&tim, NULL);
    double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
    printf("Random word vectors time: %f\n", t2-t1);
}

void
kernelUpdateParams(ParamMem_t& params, ParamMem_t& derivatives,
  ParamMem_t& adagradWeights, float learnRate) {
    timeval tim;
    gettimeofday(&tim, NULL);
    double t1=tim.tv_sec+(tim.tv_usec/1000000.0);


    unsigned int blockSize = 1024;
    unsigned int numElems = params.totalSize;
    unsigned int numBlocks = numElems / blockSize + 1;
    dim3 threadsPerBlock(blockSize, 1, 1);
    updateParams<<<numBlocks,threadsPerBlock>>>(params.base, derivatives.base,
        adagradWeights.base, learnRate, params.totalSize);
    //checkCudaErrors(cudaGetLastError());

    gettimeofday(&tim, NULL);
    double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
    printf("Update params time: %f\n", t2-t1);
}