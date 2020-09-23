## Rice Disease Detection

This was the first medium-scale capstone project of my team-mates and mine. In this project, the dataset was taken from Kaggle. It contained about 3000 images.
### Dataset:

As stated above the total images that my team and me found was 3000. There are data of 5 diseased leaves and 1 healthy rice leaf.

### Dataset Pre-processing:

I have increased the dataset using an image generator and enhanced the amount of dataset from 3000 to 8000. The techniques are:
  - Rotation: 20%
  - Zoom: 15%
  - Width Shifting: 20%
  - Height Shifting: 20%

All the data augmentation was performed using Keras's ImageDataGenerator class. Later on, normalisation and image segmentation was done on them after contour detection with edge detection.

### Model and Deployment:

After pre-processing, 3 custom CNNs, 2 Resnet32(one with Adam optimiser and another one with RADAM optimise)s were used to predict diseases/healthiness from that dataset. The accuracy was about 83% on custom CNNs and 85% on Resnets. The model was used as a worker using MQTT data/message transferring broker service via an Android application.

### Demo:
[![Demo CountPages alpha](https://j.gifs.com/k81G9K.gif)] (https://youtu.be/ePItve9IHrs)

### Support or Contact
Md. Mahmudul Haque: mahmudulhaquearfan@gmail.com
