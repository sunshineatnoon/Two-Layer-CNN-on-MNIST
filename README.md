# Two-Layer-CNN-on-MNIST

This is a two-layer convolutional neural network tested on MNIST.

The architecture is images->convolution->pooling->convolution->pooling->softmax, with cross-entropy as its cost function and weight decay.

It can reach an accurancy of 96.34%, of course different random initialization may give different result. This is just a result for one run.

To run the code, just run myCNN.m.

A Chinese version of blog about this implementation can be found at: http://www.cnblogs.com/sunshineatnoon/p/4584427.html
