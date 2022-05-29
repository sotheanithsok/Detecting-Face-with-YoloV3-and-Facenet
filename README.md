<h1 align="center" style="border: none">Detecting Face with YoloV3 and Facenet</h1>

## Overview
The goal of this experiment is to perform facial recognition on a group of people utilizing existing and well-developed technologies such as Facenet and YoloV3.

Given a set of images, the program will use the pre-trained YoloV3 model to extract faces from those images. Then, it will feed those extracted faces into the pre-trained Facenet model to map those faces onto 128 axes space.

The classification is based on some threshold of the euclidean distance between points derived from anchors or most probable images belonging to a desired group and points derived from some images.

## Architecture
![alt text](https://github.com/sotheanith/CECS-551-Facenet-Yolo/blob/master/report/download.png)

## Prerequisites
 - [Python](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [Necessary Files](https://mega.nz/folder/W2JwgZrb#3fbukdnLG308Brw5wpblMw)

## Authors
 - [Sotheanith Sok](https://github.com/sotheanith-sok "Sotheanith Sok")
 - [Alex Pahm](https://github.com/alexpham095 "Alex Pahm")
 - [Grant Chen](https://github.com/reizero01 "Grant Chen")
 
## Setup
For Windows:

    conda env create -f ./winRequirements.yml


For MacOS:

    conda env create -f ./macRequirements.yml

Run the program:

    python ./main.py --tau 1.0 --show

## Report
- [Full Report](https://github.com/sotheanith/CECS-551-Facenet-Yolo/blob/master/report/Report.pdf "Full Report")
- [Result Sheet](https://github.com/sotheanith/CECS-551-Facenet-Yolo/blob/master/report/Result.xlsx "Result Sheet")

## References
- Pre-trained YoloV3 by Thanh Nguyen: https://github.com/sthanhng/yoloface
- Pre-trained Facenet model by Hiroki Taniai: https://github.com/nyoki-mtl
- Florian Schroff, Dmitry Kalenichenko, James Philbin: FaceNet: A Unified Embedding for Face Recognition and Clustering 17 Jun 2015 (v3)
- Brownlee, Jason. “How to Develop a Face Recognition System Using FaceNet in Keras.” Machine Learning Mastery, 21 Nov. 2019, machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/.

## Course
 - [CECS 551 - Advanced Artificial Intelligence](http://catalog.csulb.edu/preview_course_nopop.php?catoid=5&coid=40041)
