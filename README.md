Multiplicative Recursive Neural Network 
===================

C++/Cuda implementation of Multiplicative Recursive Neural Network for sentiment analysis on GPU

Sources: 
http://nlp.stanford.edu/sentiment/ & http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf

#run these in CLI to configure nnvc CUDA compiler driver for osx after cuda installation:
```
export PATH=/Developer/NVIDIA/CUDA-6.5/bin:$PATH
export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-6.5/lib:$DYLD_LIBRARY_PATH
kextstat | grep -i cuda
nvcc -V
```

#to compile & run:
```
make
./network trees
```

#Sources: 
http://nlp.stanford.edu/sentiment/ & http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf

#warning:
will segfault if gpu memory is not large enough; might want to uncomment "checkCudaErrors(cudaGetLastError())" in line 79 of "kernel.cu" to stop program with "error=9" warning as opposed to attempting to run all the way through on small gpu
