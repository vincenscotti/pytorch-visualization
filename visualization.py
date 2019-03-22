import torch
import torch.nn as nn
import torchvision
import torchsummary
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

i = 0

def visualization(module):

    # We define the hook that must be run after each convolution layer is completed
    def hook(module, input, output):
        global i
        # We increase the layer count. This will be used to generate the output filename (activationXX.jpg)
        i = i + 1

        #print("INPUT : " + str(input[0].size()))
        #print("OUTPUT: " + str(output.size()))

        # Hack to check if the layer is a convolution. The convolution has always a 4-D output
        if len(output.size()) == 4:
            # We retrieve the number of output feature maps
            # output.size() has format [ BATCH x FEATURE MAPS x W x H ]
            nofmaps = output.size()[1]

            # We retrieve the output feature maps as a 3-D tensor
            ofmaps = output[0].detach().numpy()

            # Plot setup boilerplate
            actfigsize = int(math.ceil(math.sqrt(nofmaps)))
            actfig, actaxes = plt.subplots(actfigsize, actfigsize)
            plt.tight_layout()

            # For each output feature maps...
            for n in range(nofmaps):
                # ... we plot it using a blue color map...
                actaxes[n % actfigsize][n // actfigsize].matshow(ofmaps[n], cmap=plt.cm.Blues)
            print("SAVING activation" + str(i) + ".png")
            # ... then we save the plots on a file
            actfig.savefig("activation" + str(i) + ".png")
            plt.close(actfig)

        params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            #print("WEIGHT: " + str(module.weight[0][0]))
            #print("BIAS  : " + str(module.bias.size()))
            pass

    if (
        not isinstance(module, nn.Sequential)
        and not isinstance(module, nn.ModuleList)
        and not isinstance(module, nn.Linear)
    ):
        module.register_forward_hook(hook)

# Plot setup boilerplate
mpl.rcParams["figure.figsize"] = 24, 18

# We instantiate a pretrained AlexNet
model = torchvision.models.alexnet(pretrained=True)

# We print the model topology
torchsummary.summary(model, input_size=(3, 256, 256))

# We register the hook on the network model
model.apply(visualization)

#x = [torch.rand(1, *in_size).type(torch.FloatTensor) for in_size in [(3, 256, 256)]]
#model(*x)

# We preprocess the input images to match the format that was used to train AlexNet (taken from https://pytorch.org/docs/stable/torchvision/models.html)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder("./data/", torchvision.transforms.Compose([
        torchvision.transforms.Scale(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])),
    batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

# For each input image...
for data, target in test_loader:
    # ... we launch the inference
    model(data)
