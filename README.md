# pytorch-visualization

Python script to visualize intermediate convolution results.

## Requirements

Python3+ and pip3 must be installed.

With pip3 the following packages must be installed:

`pip3 install torch torchvision torchsummary matplotlib`

## Usage

Just move into the script root directory and launch:

`python3 visualization.py`

## Example output

The network topology is first printed out. The layers details are printed out. Then the inference is run and the activations are saved in the local directory. The images are named activationXX.jpg, where XX is the layer number.

```
vscotti@tiopeak2:~/Devel/pytorch-visualization$ python3 visualization.py 
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 63, 63]          23,296
              ReLU-2           [-1, 64, 63, 63]               0
         MaxPool2d-3           [-1, 64, 31, 31]               0
            Conv2d-4          [-1, 192, 31, 31]         307,392
              ReLU-5          [-1, 192, 31, 31]               0
         MaxPool2d-6          [-1, 192, 15, 15]               0
            Conv2d-7          [-1, 384, 15, 15]         663,936
              ReLU-8          [-1, 384, 15, 15]               0
            Conv2d-9          [-1, 256, 15, 15]         884,992
             ReLU-10          [-1, 256, 15, 15]               0
           Conv2d-11          [-1, 256, 15, 15]         590,080
             ReLU-12          [-1, 256, 15, 15]               0
        MaxPool2d-13            [-1, 256, 7, 7]               0
AdaptiveAvgPool2d-14            [-1, 256, 6, 6]               0
          Dropout-15                 [-1, 9216]               0
           Linear-16                 [-1, 4096]      37,752,832
             ReLU-17                 [-1, 4096]               0
          Dropout-18                 [-1, 4096]               0
           Linear-19                 [-1, 4096]      16,781,312
             ReLU-20                 [-1, 4096]               0
           Linear-21                 [-1, 1000]       4,097,000
================================================================
Total params: 61,100,840
Trainable params: 61,100,840
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 10.97
Params size (MB): 233.08
Estimated Total Size (MB): 244.80
----------------------------------------------------------------
/home/vscotti/.local/lib/python3.6/site-packages/torchvision/transforms/transforms.py:208: UserWarning: The use of the transforms.Scale transform is deprecated, please use transforms.Resize instead.
  "please use transforms.Resize instead.")
SAVING activation1.png
SAVING activation2.png
SAVING activation3.png
SAVING activation4.png
SAVING activation5.png
SAVING activation6.png
SAVING activation7.png
SAVING activation8.png
SAVING activation9.png
SAVING activation10.png
SAVING activation11.png
SAVING activation12.png
SAVING activation13.png
SAVING activation14.png
```
