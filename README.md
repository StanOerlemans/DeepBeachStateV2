# DeepBeachStateV2 - Image-Based Classification Using a Convolutional Neural Network and Transfer Learning

## Summary
This respitory contains the code that is made as an extension on previous work. We applied the algoritm for the classification of Argus imagery for the classification of single- and double-barred beach systems. We applied transfer learning, in which we fine-tuned the ResNet50 model. 

 - [Paper: Beach State Recognition Using Argus Imagery and Convolutional Neural Networks](https://www.mdpi.com/2072-4292/12/23/3953)
      - [Algoritm: DeepBeachState](https://github.com/anellenson/DeepBeachState)

## Google Colab-Pytorch Implmentation
The code is written in Google Colab using poytorch. No installments are required, you need only to change the runtime type to GPU.

## Training protocols
### Training from Scratch
When trained from scratch, the trainable parameters of the model are random initialized. This method usually requires large amount of data. 

### Transfer Learning:
Transfer learning is the improvement of learning in a new
task through the transfer of knowledge from a related task that has al-
ready been learned. While most machine learning algorithms are designed
to address single tasks, the development of algorithms that facilitate
transfer learning is a topic of ongoing interest in the machine-learning
community.

Fine tuning - starts with a pretrained model and update all of the model’s parameters for our new task, 
in essence retraining the whole model. 

Feature extraction- starts with a pretrained model and only update the final layer weights from which we derive predictions. It is called feature extraction 
because we use the pretrained CNN as a fixed feature-extractor, and only change the output layer.

 ## pre-trained models
 In the algorithm there is a choice for several pre-trained models. These include:
 ### 1- AlexNet (2012)
In 2012, AlexNet significantly outperformed all the prior competitors and won the challenge by reducing the top-5 error from 26% to 15.3%.
The second place top-5 error rate, which was not a CNN variation, was around 26.2%. see [AlexNet document](http://cvml.ist.ac.at/courses/DLWT_W17/material/AlexNet.pdf)
### VGGNet (2014)
The runner-up at the ILSVRC 2014 competition is dubbed VGGNet by the community and was developed by Simonyan and Zisserman . VGGNet consists of 16 convolutional layers and is very appealing because of its very uniform architecture. 
Similar to AlexNet, only 3x3 convolutions, but lots of filters. Trained on 4 GPUs for 2–3 weeks. see [VGGNet document](https://arxiv.org/pdf/1409.1556.pdf)
### ResNet(2015)
ILSVRC 2015, the so-called Residual Neural Network (ResNet) by Kaiming He et al introduced 
anovel architecture with “skip connections” and features heavy batch normalization.They were able to train a NN with 152 layers while still having lower complexity than VGGNet. 
It achieves a top-5 error rate of 3.57% which beats human-level performance on this dataset. see [ResNet document](https://arxiv.org/abs/1512.03385)

## Work Flow

Step 0) Open the code in Google Colab and set runtime type to GPU. 

Step 1) Collect the data and upload it to your Google Drive. Run the code and visualize your data.

Step 2) Initialize the network

Step 3) Data preparation

Before training we need to prepare our data:

- Distribute data into train, validation and test sets
- Meet model requirements in terms of image size and pixel range
- Apply augmentations
- Apply weights

Step 4) Train the network

This will output training information that looks like this:

<pre><code>
model: resnet50, Epoch: [48/1000]
----------
train Loss: 0.2437 Acc: 0.9118 LR:0.00040297376309494926
val Loss: 0.6593 Acc: 0.7353 LR:0.00040297376309494926
EarlyStopping counter: 20 out of 20

Early stopping
Training complete in 5m 57s
Best val Acc: 85.294118
</code></pre>

The training and validation losses can be plotted, looking like this:

![](/figures/loss_plot_resnet50_ft_1.png?raw=true)

Step 5) Test the models

Testing the model on test data results in:

<pre><code>
Accuracy of the network on the test dataset : 80 %
</code></pre>

Step 6) Report the performance

An important part in machine learning is the skill evaluation which will be done in several ways:
- Global skill, F1
- Precision
- Recall
- Accuracy 
- Normalized mutual info
- Confusion matrix

<pre><code>
              precision    recall  f1-score   support

         Ref       1.00      0.90      0.95        10
         LTT       0.74      0.88      0.80        16
         TBR       0.79      0.73      0.76        15
         RBB       0.88      0.74      0.80        19
         LBT       0.70      0.88      0.78         8

    accuracy                           0.81        68
   macro avg       0.82      0.82      0.82        68
weighted avg       0.82      0.81      0.81        68
</code></pre>

And the confusion matrix:

![](/figures/loss_plot_resnet50_ft_1.png?raw=true)

Step 7) Model Visualization

We applied model visualization for interpretation.

[Class Activation Map methods implemented in Pytorch](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
 ## References
 
   1-[transfer learning tutorial on pytorch](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
   
   2- [transfer learning tutorial](https://ruder.io/transfer-learning/)
   
   3- [implemnted CNN overview](https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5)
   
   4- [transfer learning tutorial on github](http://cs231n.github.io/transfer-learning/)
