# neural-networks-scala
Scala port of Michael Nielsen's book ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com)

I'm building this port as I study the book, so it's not yet complete.

To setup the MNIST data as per the book's train, validation and test sets, create a top-level `data` directory, and download the following four data files into it from http://yann.lecun.com/exdb/mnist/
* [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
* [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
* [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
* [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)

Run the following commands at the command line:
```
cp data/t10k-images-idx3-ubyte.gz data/mnist-test-images-10k.gz
cp data/t10k-labels-idx1-ubyte.gz data/mnist-test-labels-10k.gz
```

In SBT run the following commands:
```
runMain neuralNetworks.MNISTDataSlicer data/train-images-idx3-ubyte.gz data/train-labels-idx1-ubyte.gz data/mnist-train-images-50k.gz data/mnist-train-labels-50k.gz 0 50000
runMain neuralNetworks.MNISTDataSlicer data/train-images-idx3-ubyte.gz data/train-labels-idx1-ubyte.gz data/mnist-validate-images-10k.gz data/mnist-validate-labels-10k.gz 50000 10000
```

Your `data` directory should now contain the following files:
```
mnist-test-images-10k.gz
mnist-test-labels-10k.gz
mnist-train-images-50k.gz
mnist-train-labels-50k.gz
mnist-validate-images-10k.gz
mnist-validate-labels-10k.gz
```
