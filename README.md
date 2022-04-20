# 2C2-ESRGAN

ESRGAN (Enhanced Super Resolution GAN) using two 2x2 (kernel size) conv2d layers instead of a traditional single 3x3 conv2d layer in its conv block.

This idea is inspired by [this reddit post](https://www.reddit.com/r/MachineLearning/comments/u2vim0/d_replacing_3x3_convolutions_with_two_2x2/) which suggested using two 2x2 conv2d layers instead of one 3x conv2d layer when making neural network architectures. I wanted to see if this would work for Super Resolution, so I modified ESRGAN to use this type of conv block.

I did make a few small changes from the suggested implementation. First, I moved the activation layer to the end of the sequence instead of between the conv layers. This was to make it more similar to how ESRGAN natively does it. I also swapped the order of the conv layers, as it was doing padding=0 first, then padding=1, which was causing data loss around the edges of the image. Doing padding=1 first ensures that the iamge is expanded from the padding before being shrunk down to the normal size again.

The end result of this is an architure that performs the same yet is roughly 55% the size when saved as a .pth file, yet also twice as deep.

This repo just contains the python files for the architecture used as well as the pretrained model I created. If you would like to perform inference with this architecture, I will be adding support to [my ESRGAN fork](https://github.com/joeyballentine/esrgan).
