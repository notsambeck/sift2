SIFT is a visual experiment. It has two parts:

1. SIFT systematically generates every possible image
2. A neural network evaluates each image to determine if it is interesting or not. Candidate images are displayed and/or saved to file.

The goal is to answer a question - using brute-force combinatorics, how many images would you have to generate to find one that is indistinguishable from a photograph?

At the beginning of this project (in 2009), it seemed that the needed number of images was impossible. Or, more precisely, that it was ridiculously unlikely that any generated image would have properties we associate with real images.

Given the scale of the numbers involved, the best strategy for answering this question is to start with ‘minimal’ images - the smallest data objects that start to take on the properties of images. Fortunately, with the development of machine learning as a viable technology over the last ~10 years, scientists have started researching closely related questions. One relevant result comes from CIFAR, an academic project that studies computer vision. They have shown that scene recognition is possible with very small images, both by computers and by people. A 32x32 pixel image contains enough information to identify the context of ~90% of cases. This applies to both humans and computers.


SIFT is built from a bunch of pre-existing tools, which can be found in requirements.txt. It should not be dependent on CUDA, but it will benefit from CUDA / GPU acceleration.

Install the requirements (ideally in a virtualenv) and then run either of these in Python 3:
python sift.py
python sift-nonvisual.py
