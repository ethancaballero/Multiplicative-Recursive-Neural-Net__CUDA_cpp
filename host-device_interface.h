#ifndef HOST_DEVICE_INTERFACE_H
#define HOST_DEVICE_INTERFACE_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdlib>

#include "helper_cuda.h"
#include "kernels.h"

class CudaInterface {
  public:
    static void initialize();
    static void cleanup();

    /***** Host/Device Memory Management *****/
    static void allocMem(float** mem, unsigned int size, bool device);
    static void freeMem(float* mem, bool device);
    static void transferMem(ParamMem_t pmem1, ParamMem_t pmem2, bool device1, bool device2);

    static void allocParamMem(ParamMem_t& pmem, unsigned wordVecDim, unsigned nClasses, unsigned nWords, bool device);
    static void freeParamMem(ParamMem_t& pmem);
    static void fillParamMem(ParamMem_t& pmem, int byteVal);

    /***** CUBLAS wrappers *****/
    static int cublasMatrixMult(float* A, float* B, float* C, MatrixSize_t mSize);

    /***** Unused *****/
    CudaInterface() {}
 
  private:
    static const float mAlpha;
    static const float mBeta;
    static int mDevID;
    static cudaDeviceProp mDeviceProperty;
    static cublasHandle_t mCublasHandle;
};

#endif // HOST_DEVICE_INTERFACE_H
