#ifndef KERNELS_H
#define KERNELS_H

#include <cstdlib>
#include <cstring>

typedef struct matrixSize {
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} MatrixSize_t;

typedef struct mirrorMem {
    float* host;
    float* device;
    unsigned int size;
} MirrorMem_t;

/* encapsulates the model parameters
   allocates contiguous memory on device starts
   remaining pointers point to start of their matrices*/
typedef struct paramMem {
    float* base;
    float* softmax;
    float* transformW;
    float* transformV;
    float* wordVecs;
    unsigned int totalSize;
    unsigned int wordVecDim;
    unsigned int nClasses;
    unsigned int nWords;
    bool device;
} ParamMem_t;

/***** Kernel wrappers *****/
void kernelRandomwordVecs(ParamMem_t& params, float threshold);
void kernelUpdateParams(ParamMem_t& params, ParamMem_t& derivatives,
  ParamMem_t& adagradWeights, float learnRate);

#endif // KERNELS_H