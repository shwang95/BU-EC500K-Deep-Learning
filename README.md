# BU-EC500K-Deep-Learning

* Textbook: https://github.com/HFTrader/DeepLearningBook

## Library Versions

* Tensorflow: 1.2.0

* Cuda: 8.0

* Cudnn: 5.1

* Python: 3.6

* Anaconda: 3-5.1.0

## Install tensorflow with GPU support

1. Install Anaconda3 with "add enviroment path" checked

2. Install latest Nvidia driver

3. Check ur GPU supported cudnn version first at: https://developer.nvidia.com/cuda-gpus

3. Install Cuda to Match the Cudnn version

4. Extract Cudnn, and put it under ~/CUDA_PATH/vX.0/

5. Check tensorflow release log to find the highest Tensorflow version support your Python, Cuda and Cudnn at: https://github.com/tensorflow/tensorflow/releases?after=v1.3.0-rc0

6. pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.X.0-cp3X-cp3Xm-win_amd64.whl

7. Test tensorflow installation with this script: https://gist.github.com/mrry/ee5dbcfdd045fa48a27d56664411d41c

Example:

| GPU | Cuda | Cudnn | Python | Tensorflow |
| --- | ---- | ----- | ------ | ---------- |
| 980M | 8.0 | 5.1 | 3.6 | 1.2.0 |
| 1070 | 8.0 | 6.0 | 3.6 | 1.4.0 |

## Test installation:
*  import tensorflow as tf
*  hello = tf.constant('Hello, TensorFlow!')
*  sess = tf.Session()
*  print(sess.run(hello))
