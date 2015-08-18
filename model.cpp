#include "model.h"

#include <cstdlib>
#include <algorithm>
#include <fstream>

#include "host-device_interface.h"

Tree::Tree(const std::string& treeString) {
    Tree();
    int stringLength = treeString.size();
    if (stringLength < 4) {      
        std::cout << "malformed" << treeString.c_str() << "\n";
        std::exit(1);
    }

    if (!(treeString[0] == '(' && treeString[stringLength-1] == ')')) {        
        std::cout << "error reading" << treeString.c_str() << "\n";
        std::exit(1);
    }

    mGoldStandardClass = std::atoi(treeString.substr(1, 2).c_str());
    if (treeString[3] != '(') {
        mCheckLeaf = true;
        mLabel = treeString.substr(3, stringLength-4);
        std::transform(mLabel.begin(), mLabel.end(), mLabel.begin(), ::tolower);
    } else {
        mCheckLeaf = false;
        int splitIdx = -1;
        int currentDepth = 1;
        for (int i = 4; i < stringLength; i++) {
            if (treeString[i] == '(') currentDepth++;
            if (treeString[i] == ')') currentDepth--;
            if (currentDepth == 0) {
                splitIdx = i+1;
                break;
            }
        }
        if (!(splitIdx > 3 && splitIdx < stringLength-2)) {
            std::cout << "error parsing" << treeString.c_str() << "\n";
            std::exit(1);
        }
        mLChild = new Tree(treeString.substr(3, splitIdx-3));
        mRChild = new Tree(treeString.substr(splitIdx+1, stringLength-splitIdx-2));
    }
}

void
Tree::getLeaves(std::set<std::string>& words) {
    if (mCheckLeaf) {
        if (!mLabel.empty()) {
            words.insert(mLabel);
        }
    } else {
        if (mLChild != NULL) mLChild->getLeaves(words);
        if (mRChild != NULL) mRChild->getLeaves(words);
    }
}

void
Tree::assignNodeVecsAndTag(const std::map<std::string, int>& wordToTags, int unknownWordTag,
    unsigned int wordVecDim, unsigned int nClasses) {
    if (mCheckLeaf) {
        std::map<std::string, int>::const_iterator it = wordToTags.find(mLabel);
        if (it != wordToTags.end()) {
            mWordTag = it->second;
        } else {
            mWordTag = unknownWordTag;
        }
    } else {
        if (mLChild != NULL)
            mLChild->assignNodeVecsAndTag(wordToTags, unknownWordTag, wordVecDim, nClasses);
        if (mRChild != NULL)
            mRChild->assignNodeVecsAndTag(wordToTags, unknownWordTag, wordVecDim, nClasses);
    }   
}

void
Tree::cleanUp() {
    if (mLChild != NULL) {
        mLChild->cleanUp();
        mLChild = NULL;
    }
    if (mRChild != NULL) {
        mRChild->cleanUp();
        mRChild = NULL;
    }
}

void
Tree::read(std::vector<Tree*>& trees, const std::string& path) {
    std::ifstream fs(path.c_str());
    std::string line;
    while(std::getline(fs, line)) {
        trees.push_back(new Tree(line));
    }
}

void
Tree::getLeaves(std::vector<Tree*>& trees, std::set<std::string>& words) {
    for (int i = 0; i < trees.size(); i++) {
        trees[i]->getLeaves(words);
    }
}

void
Tree::assignAllNodeVecsAndTag(std::vector<Tree*>& trees, const std::map<std::string, int>& wordToTags,
  int unknownWordTag, unsigned int wordVecDim, unsigned int nClasses) {
    for (int i = 0; i < trees.size(); i++) {
        trees[i]->assignNodeVecsAndTag(wordToTags, unknownWordTag, wordVecDim, nClasses);
    }
}

void
Tree::cleanTrees(std::vector<Tree*>& trees) {
    for (int i = 0; i < trees.size(); i++) {
        trees[i]->cleanUp();
    }
}
