# DeepBeachStateV2 - Image-Based Classification Using a Convolutional Neural Network and Transfer Learning

## Summary
This respitory contains the code that is made as an extension on [previous work](https://github.com/anellenson/DeepBeachState). We applied the algoritm for the classification of Argus imagery for the classification of single- and double-barred beach systems. We applied transfer learning, in which we fine-tuned the ResNet50 model. 

## Google Colab-Pytorch Implmentation
The code is written in Google Colab using poytorch. No installments are required, you need only to change the runtime type to GPU.

## Work Flow

Step 0) Open the code in Google Colab and set runtime type to GPU 

Step 1) Collect the data and upload it to your Google Drive. Run the code and visualize your data

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
model: resnet50, Epoch: [51/1000]
----------
train Loss: 0.1982 Acc: 0.9228 LR:0.00040462982701110727
val Loss: 0.5292 Acc: 0.8529 LR:0.00040462982701110727
EarlyStopping counter: 20 out of 20

Early stopping
Training complete in 6m 23s
Best val Acc: 89.705882
</code></pre>

The training and validation losses can be plotted, looking like this:

![](/figures/loss_plot_ft_resnet50_1.png)

Step 5) Test the models

Testing the model on test data results in:

<pre><code>
Accuracy of the network on the test dataset : 80%
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

         Ref       0.71      1.00      0.83        10
         LTT       0.79      0.69      0.73        16
         TBR       0.76      0.87      0.81        15
         RBB       0.92      0.67      0.77        18
         LBT       0.80      0.89      0.84         9

    accuracy                           0.79        68
   macro avg       0.80      0.82      0.80        68
weighted avg       0.81      0.79      0.79        68
</code></pre>

And the confusion matrix:

![](/figures/CM_ft_resnet50_1.png)

Step 7) Model Visualization

We applied model visualization by Class Acitivation Mapping following: 

[Class Activation Map methods implemented in Pytorch](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

![](/CAM/gradcam++_original_Example_img_1.jpg) ![](/CAM/gradcam++_cam_Example_img_1.jpg) ![](/CAM/gradcam++_cam_gb_Example_img_1.jpg)

 ## References
 
   1 - [transfer learning tutorial on pytorch](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
   
   2 - [Beach State Recognition Using Argus Imagery and Convolutional Neural Networks](https://www.mdpi.com/2072-4292/12/23/3953)
