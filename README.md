# BU-EC500K-Deep-Learning

* Textbook: https://github.com/HFTrader/DeepLearningBook

* Windows Tensorflow GPU install: 
  * Install Cuda 7.5
  * Install CuDnn 5.1 (not 6.0), put it under Cuda folder and add installation folder to system env path
  * Install MSVCP140.DLL:
    * https://www.microsoft.com/en-gb/download/details.aspx?id=48145
  * "pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.2.0-cp36-cp36m-win_amd64.whl"
  * Test installation:
    *  import tensorflow as tf
    *  hello = tf.constant('Hello, TensorFlow!')
    *  sess = tf.Session()
    *  print(sess.run(hello))
