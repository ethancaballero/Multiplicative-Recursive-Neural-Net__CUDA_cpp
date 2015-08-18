#ifndef TRAIN_H
#define TRAIN_H

#include <vector>
#include <string>

#include "host-device_interface.h"

#include "model.h"

typedef struct trainOptions {
  std::string trainingPath;
  std::string developmentPath;
  int batchSize;
  int nCycles;
  float learnRate;
  float rSoftmax;
  float rTransformW;
  float rTransformV;
  float rwordVecs;
} sliderOptions_t;

typedef struct sentimentModel {
    int wordVecDim; // D
    int nClasses; // C
    int nWords; // L
    std::map<std::string, int> wordToTag;

    ParamMem_t parameters_h; // parameters in host memory
    ParamMem_t parameters_d; // parameters in device memory
    ParamMem_t derivs_d; // derivatives in device memory
    ParamMem_t adagradWeights_d; // weights in device memory

    float* tempNodeClassDist_d; // temp node class distribution in device memory
    float* nodeVec_d;    // temp node vector in device memory
} sModel_t;

class SentimentTrain {
  public:
    SentimentTrain(sliderOptions_t options, int wordVecDim, int nClasses);
    void train();

  private:
    void initNodeVecs();
    void trainBatch(int startIdx, int endIdx);
    float computeDerivs(int startIdx, int endIdx);
    void forwardProp(Tree* tree);
    void backProp(Tree* tree);

    std::vector<Tree*> mTrainingTree;
    std::vector<Tree*> mDevelopmentTree;
    sModel_t mMod;
    sliderOptions_t mOpts;

    static const std::string UNKNOWN_WORD;
};

#endif // TRAIN_H
