# Detecting Face with YoloV3 and Facenet
## Members
- [Sotheanith Sok](https://github.com/sotheanith "Sotheanith Sok")
- [Alex Pahm](https://github.com/alexpham095 "Alex Pahm")
- [Grant Chen](https://github.com/reizero01 "Grant Chen")

## Goal
The goal of this experiment is to perform facial recognition on a group of people utilizing existing and well-developed technologies such as Facenet and YoloV3.

Given a set of images, the program will use the pre-trained YoloV3 model to extract faces from those images. Then, it will feed those extracted faces into the pre-trained Facenet model to map those faces onto 128 axes space.

The classification is based on some threshold of the euclidean distance between points derived from anchors or most probable images belonging to a desired group and points derived from some images.

## Architecture
![alt text](https://drive.google.com/uc?export=view&id=1DiDWO93Tgk1UjZXoTn40tt_PMRakdlex)

## Prerequisites
Due to the complex nature of getting Tensorflow-GPU and various other libraries running in the same environment, it is highly recommended to install packages using Anaconda. 

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

