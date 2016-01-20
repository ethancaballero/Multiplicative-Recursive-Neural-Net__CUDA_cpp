#include <cstdlib>
#include <string>

#include "train.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("provide directory of training file(s)\n");
        std::exit(1);
    }
    std::string treeDirectory(argv[1]);
    std::string trainingPath = treeDirectory + "/train.txt";
    std::string developmentPath = treeDirectory + "/dev.txt";
    
    // model parameters
    int wordVecDim = 32;  // Word Vector Dimensions
    int nClasses = 5;   // number of Sentiment Classes

    // training parameters
    sliderOptions_t options = {
        trainingPath,
        developmentPath,
        25,
        2,
        0.01,
        0.0001,
        0.001,
        0.001,
        0.0001
    };

    SentimentTrain trainer(options, wordVecDim, nClasses);
    trainer.train();

    return 0;
}
