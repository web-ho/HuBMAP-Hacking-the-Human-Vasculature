# About Repo:

In this repo I have shared my approach to train a model that detects and segments instances of microvascular structures on a given human kidney tissue slides.
You can read more about the problem and data here <a href="https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/overview">LINK</a>.

# Data Description:

The solution that got me my first bronze on kaggle.

As described on the competition site-
The data comprises tiles extracted from five Whole Slide Images (WSI) split into two datasets. Tiles from Dataset 1 have annotations that have been expert reviewed. Dataset 2 comprises the remaining tiles from these same WSIs and contain sparse annotations that have not been expert reviewed.

- All of the test set tiles are from Dataset 1.
- Two of the WSIs make up the training set, two WSIs make up the public test set, and one WSI makes up the private test set.
- The training data includes Dataset 2 tiles from the public test WSI, but not from the private test WSI.

They have also provided, as Dataset 3, tiles extracted from an additional nine WSIs. These tiles have not been annotated. Thus, we could use semi- or self-supervised learning techniques on this data.

Also, important to remember that we only have to predict blood_vessels for the submission to work.

# Method:

My approach was to train a model using both dataset 1 and 2. I used Yolov8 to set up a simple baseline and it performed amazingly.

- simple train-test split with validation data from dataset 2.
- img_size = 512
- epochs = 100
- optimizer = adam
- learning_rate = 0.0001
- batch_szie = 4
- iou = 0.1
rest of the hyperparameters were default.

For Inference-

- img_size = 512
- conf = 0.01
- iou = 0.6

# Another approach:

I tried to implement the top solutions of sartorious cell segmentation task to this task, due to timeline and resources issues couldn't make it work-

Two stage approach-
First stage:
- trained five yolov8 models, 3 on 512 img_size, 1 on 720 img_size and 1 on 1024 img_size. 
- three models with 512 img_size were trained using different hyperparmeters. 
- all the five models were ensembled using WBF.

Second stage:
- a unetplusplus model was trained on the cropped and padded bbox images to predict mask.
- model was trained with img_size=256 and 128.
- cropped images using mask coordinates and this caused the model to perform very poorly.



