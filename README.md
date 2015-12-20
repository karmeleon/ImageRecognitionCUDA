# ImageRecognitionCUDA
Final project for ECE 408 at UIUC

What it is
----------
It's a CUDA-accelerated image recognition training program. Specifically, given a set of input images and corresponding labels, it uses CUDA to find all the significant [Haar-like features](https://en.wikipedia.org/wiki/Haar-like_features) in all the images, then uses AdaBoost to identify the features that most accurately describe the object being trained.

Running
-------
Open VSProject/408Final.v12.suo in Visual Studio 2013 and run. CUDA 7.5 doesn't seem to support VS2015 yet. It uses images from the [MNIST handwritten digit sample set](http://yann.lecun.com/exdb/mnist/) to create a strong classifier for the digit '0'. Note that this was done for an ECE408 project, and the actual identifier for the created strong classifier was not part of the project scope. As a result, you can't actually use it to identify digits.

Requirements
------------
* Visual Studio 2013
* 64-bit Windows 7+
* CUDA 7.5 Toolkit
* A CUDA-capable card with 1.5 GB+ of VRAM. The VS project generates code for compute capabilities 3.0 and 5.2, but you can easily edit the project settings to generate more.
* At least 8 GB of system RAM
