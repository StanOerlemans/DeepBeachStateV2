# DeepBeachStateV2

This respitory provides a skeleten of how to implement a CNN on Argus data.

It is an extension on previous work for the automated classification of not only single-barred beach state, but also for double-barred beach states.

The code is written in Google Colab so no installments are required, you need only to change the runtime type to GPU.

## Work Flow

Step 0) Open the code in Google Colab and set runtime type to GPU. 

Step 1) Collect the data and upload it to your Google Drive. Run the code and visualize your data.

Step 2) Initialize the network

Step 3) Train the network

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

Step 4) Test the models

