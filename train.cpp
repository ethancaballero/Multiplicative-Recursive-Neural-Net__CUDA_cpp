#include "train.h"

#include <algorithm>
#include <unistd.h>

#include "kernels.h"

const std::string SentimentTrain::UNKNOWN_WORD = "<unknown>";

SentimentTrain::SentimentTrain(sliderOptions_t options, int wordVecDim, int nClasses) {
    mOpts = options;
    mMod.wordVecDim = wordVecDim;
    mMod.nClasses = nClasses;
}

void
SentimentTrain::initNodeVecs() {
    std::set<std::string> words;
    Tree::getLeaves(mTrainingTree, words);
    words.insert(UNKNOWN_WORD);
    int wordTag = 0;
    std::set<std::string>::iterator it;
    for (it = words.begin(); it != words.end(); ++it) {
        mMod.wordToTag[*it] = wordTag;
        wordTag++;
    }
    mMod.nWords = wordTag;
    std::cout << "# of train trees: " << mTrainingTree.size() << ", " << "# of words: " << mMod.nWords << "\n";
    Tree::assignAllNodeVecsAndTag(mTrainingTree, mMod.wordToTag, mMod.wordToTag[UNKNOWN_WORD],
      mMod.wordVecDim, mMod.nClasses);
}

void
SentimentTrain::train() {
    CudaInterface::initialize();
    Tree::read(mTrainingTree, mOpts.trainingPath);
    Tree::read(mDevelopmentTree, mOpts.developmentPath);

    int nBatches = mTrainingTree.size() / mOpts.batchSize + 1;

    initNodeVecs();

    CudaInterface::allocParamMem(mMod.parameters_d, mMod.wordVecDim,
        mMod.nClasses, mMod.nWords, true);

    CudaInterface::allocParamMem(mMod.parameters_h, mMod.wordVecDim,
        mMod.nClasses, mMod.nWords, false);

    CudaInterface::allocParamMem(mMod.derivs_d, mMod.wordVecDim,
        mMod.nClasses, mMod.nWords, true);

    CudaInterface::allocParamMem(mMod.adagradWeights_d, mMod.wordVecDim,
        mMod.nClasses, mMod.nWords, true);

    CudaInterface::allocMem(&(mMod.tempNodeClassDist_d), mMod.nClasses, true);
    CudaInterface::allocMem(&(mMod.nodeVec_d), mMod.wordVecDim, true);

    CudaInterface::fillParamMem(mMod.parameters_d, 0);
    kernelRandomwordVecs(mMod.parameters_d, 0.001);

    for (unsigned cycle = 0; cycle < mOpts.nCycles; cycle++) {
        std::random_shuffle(mTrainingTree.begin(), mTrainingTree.end());

        CudaInterface::fillParamMem(mMod.adagradWeights_d, 0);

        for (unsigned batch = 0; batch < nBatches; batch++) {
            int startIdx =  batch * mOpts.batchSize;
            int endIdx   = (batch+1) * mOpts.batchSize;
            if (startIdx >= mTrainingTree.size()) break;
            if (endIdx + mOpts.batchSize > mTrainingTree.size()) endIdx = mTrainingTree.size();

            std::cout << "Cycle: " << cycle << " Batch: " << batch << "\n";
            trainBatch(startIdx, endIdx);
        }
    }

    CudaInterface::freeParamMem(mMod.parameters_d);
    CudaInterface::freeParamMem(mMod.parameters_h);
    CudaInterface::freeParamMem(mMod.derivs_d);
    CudaInterface::freeParamMem(mMod.adagradWeights_d);
    CudaInterface::freeMem(mMod.tempNodeClassDist_d, true);
    CudaInterface::freeMem(mMod.nodeVec_d, true);
    Tree::cleanTrees(mTrainingTree);
    Tree::cleanTrees(mDevelopmentTree);
    CudaInterface::cleanup();
}

void
SentimentTrain::trainBatch(int startIdx, int endIdx) {
    float value = computeDerivs(startIdx, endIdx);
    kernelUpdateParams(mMod.parameters_d, mMod.derivs_d, mMod.adagradWeights_d, mOpts.learnRate);
}

//learning_rate, weight_updates, & derivatives are in kernel.cu
void
SentimentTrain::forwardProp(Tree* tree) {    
}

void
SentimentTrain::backProp(Tree* tree) {    
}

float
SentimentTrain::computeDerivs(int startIdx, int endIdx) {
    CudaInterface::fillParamMem(mMod.derivs_d, 0);

    for (int i = startIdx; i < endIdx; i++) {
        forwardProp(mTrainingTree[i]);
    }

    for (int i = startIdx; i < endIdx; i++) {
        backProp(mTrainingTree[i]);
    }

    return 1.0;
}