#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <set>
#include <map>
#include <iostream>

class Tree {
  public:
    std::string mLabel;
    bool mCheckLeaf;
    int mGoldStandardClass;
    int mPredictClass;
    float* mPredictClassDist_d;

    int mWordTag;
    float* mNodeVec_d; // this is stored in device memory

    Tree* mLChild;
    Tree* mRChild;

    Tree() {
        mLChild = NULL;
        mRChild = NULL;
        mPredictClassDist_d = NULL;
        mNodeVec_d = NULL;
        mWordTag = -1;
    }

    Tree(const std::string& treeString);

    void getLeaves(std::set<std::string>& words);
    void assignNodeVecsAndTag(const std::map<std::string, int>& wordToTags, int unknownWordTag,
      unsigned int wordVecDim, unsigned int nClasses);
    void cleanUp();

    static void read(std::vector<Tree*>& trees, const std::string& path);
    static void getLeaves(std::vector<Tree*>& trees, std::set<std::string>& words);
    static void assignAllNodeVecsAndTag(std::vector<Tree*>& trees, const std::map<std::string, int>& wordToTags,
      int unknownWordTag, unsigned int wordVecDim, unsigned int nClasses);
    static void cleanTrees(std::vector<Tree*>& trees);
};

#endif // MODEL_H
