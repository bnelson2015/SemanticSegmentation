# SemanticSegmentation

This code is a result of following the tutorial outlined at:
https://www.mathworks.com/help/images/multispectral-semantic-segmentation-using-deep-learning.html

The main MATLAB file is SegmentationTut.m. The other files in the directory are
for checking GPU support and loading a pre-trained U-Net model. matRead.m is a
helper function to load images from .mat.

If you are planning on running this on your own computer, make sure to change the
imageDir variable on line three to a directory on the local machine.
