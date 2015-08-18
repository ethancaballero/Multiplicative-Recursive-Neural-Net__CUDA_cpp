#include "host-device_interface.h"

#include <cstring>
#include <iostream>

/*******************************/
/***** Interface Functions *****/
/*******************************/

const float CudaInterface::mAlpha = 1.0f;
const float CudaInterface::mBeta  = 0.0f;
int CudaInterface::mDevID;
cudaDeviceProp CudaInterface::mDeviceProperty;
cublasHandle_t CudaInterface::mCublasHandle;

void
CudaInterface::initialize() {
    mDevID = 0;
    checkCudaErrors(cudaSetDevice(mDevID));
    checkCudaErrors(cudaGetDevice(&mDevID));
    checkCudaErrors(cudaGetDeviceProperties(&mDeviceProperty, mDevID));
    checkCudaErrors(cublasCreate(&mCublasHandle));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", mDevID, mDeviceProperty.name, mDeviceProperty.major, mDeviceProperty.minor);

    // needs a larger block size for Fermi and above
    int block_size = (mDeviceProperty.major < 2) ? 16 : 32;
}

void
CudaInterface::cleanup() {
    checkCudaErrors(cublasDestroy(mCublasHandle));
}

/*****************************/
/***** Memory Management *****/
/*****************************/

void
CudaInterface::allocMem(float** mem, unsigned int size, bool device) {
    if (device)
        checkCudaErrors(cudaMalloc((void **) mem, size * sizeof(float)));
    else
        *mem = (float*) malloc(size * sizeof(float));
}

void
CudaInterface::freeMem(float* mem, bool device) {
    if (device)
        checkCudaErrors(cudaFree(mem));
    else
        free(mem);
}

void
CudaInterface::transferMem(ParamMem_t pmem1, ParamMem_t pmem2, bool device1, bool device2) {
    cudaMemcpyKind kind;
    if (device1 && device2) kind = cudaMemcpyDeviceToDevice;
    else if (!device1 && device2) kind = cudaMemcpyDeviceToHost;
    else if (device1 && !device2) kind = cudaMemcpyHostToDevice;
    else  kind = cudaMemcpyHostToHost;

    checkCudaErrors(cudaMemcpy(pmem1.base, pmem2.base, pmem1.totalSize * sizeof(float), kind));
}

void
CudaInterface::allocParamMem(ParamMem_t& pmem, unsigned wordVecDim, unsigned nClasses, unsigned nWords, bool device) {
    checkCudaErrors(cudaSetDevice(mDevID));
    checkCudaErrors(cudaGetDevice(&mDevID));
    pmem.wordVecDim = wordVecDim;
    pmem.nClasses = nClasses;
    pmem.nWords = nWords;
    pmem.device = device;
    pmem.totalSize  = nClasses * (wordVecDim+1);
    pmem.totalSize += wordVecDim * (2*wordVecDim+1);
    pmem.totalSize += 4 * wordVecDim * wordVecDim * wordVecDim;
    pmem.totalSize += nWords * wordVecDim; // 5*33+32*65+4*1024*32+16582*32
    if (device) {
        cudaEvent_t sync_event;
        checkCudaErrors(cudaMalloc((void **) &(pmem.base), pmem.totalSize * sizeof(float)));
    } else {
        pmem.base = (float*) malloc(pmem.totalSize * sizeof(float));
    }
    std::cout << "allocated " << pmem.totalSize * sizeof(float) << " bytes to " <<  pmem.base << "\n";
    pmem.softmax = pmem.base;
    pmem.transformW = pmem.softmax + nClasses * (wordVecDim+1);
    pmem.transformV = pmem.transformW + nClasses * (2*wordVecDim+1);
    pmem.wordVecs = pmem.transformV + 4 * wordVecDim * wordVecDim * wordVecDim;
}

void
CudaInterface::freeParamMem(ParamMem_t& pmem) {
    checkCudaErrors(cudaSetDevice(mDevID));
    checkCudaErrors(cudaGetDevice(&mDevID));
    if (pmem.device)
        checkCudaErrors(cudaFree(pmem.base));
    else
        free(pmem.base);
    pmem.base = NULL;
    pmem.softmax = NULL;
    pmem.transformW = NULL;
    pmem.transformV = NULL;
    pmem.wordVecs = NULL;
}

void
CudaInterface::fillParamMem(ParamMem_t& pmem, int byteVal) {
    checkCudaErrors(cudaSetDevice(mDevID));
    checkCudaErrors(cudaGetDevice(&mDevID));
    std::cout << "  setting " << pmem.totalSize * sizeof(float) << " bytes to " <<  pmem.base << "\n";
    if (pmem.device) {
        checkCudaErrors(cudaThreadSynchronize());
        checkCudaErrors(cudaMemset(pmem.base, byteVal, pmem.totalSize * sizeof(float)));
        checkCudaErrors(cudaThreadSynchronize());
    } else
        memset(pmem.base, byteVal, pmem.totalSize * sizeof(float));
}

/***************************/
/***** CUBLAS Wrappers *****/
/***************************/

int
CudaInterface::cublasMatrixMult(float* A, float* B, float* C, MatrixSize_t mSize) {
    checkCudaErrors(cudaSetDevice(mDevID));
    checkCudaErrors(cudaGetDevice(&mDevID));
    checkCudaErrors(cublasSgemm(mCublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, mSize.uiWB, mSize.uiHA, 
                                mSize.uiWA, &mAlpha, B, mSize.uiWB, A, mSize.uiWA, &mBeta, C, mSize.uiWA));
    return 0;
}
