# SIFT

### SIFT is a visual experiment. It has two parts:

1. SIFT systematically generates every possible image
2. A neural network evaluates each image to determine if it is interesting or not. Candidate images are displayed and/or saved to file.

### The goal is to answer a question:
### Using brute-force combinatorics, how many images would you have to generate to find one that is indistinguishable from a photograph?

At the beginning of this project (in 2009), it seemed that the needed number of images was impossible. Or, more precisely, that it was ridiculously unlikely that any generated image would have properties we associate with real images.

Given the scale of the numbers involved, the best strategy for answering this question is to start with ‘minimal’ images - the smallest data objects that start to take on the properties of images. Fortunately, with the development of machine learning as a viable technology over the last ~10 years, scientists have started researching closely related questions. One relevant result comes from CIFAR, an academic project that studies computer vision. They have shown that scene recognition is possible with very small images, both by computers and by people. A 32x32 pixel image contains enough information to identify the context of ~90% of cases. This applies to both humans and computers.


SIFT is built with many pre-existing tools, which can be found in requirements.txt. It is not dependent on CUDA, but it will benefit from CUDA / GPU acceleration.

# INSTALLATION for Linux 16
(If you are using older Linux or most other OS, you will probably have to specify Python3 and PIP3.)

1. Install python3, git (and optionally virtualenv + virtualenvwrapper) globally

2. Requirements.txt lists tensorflow-gpu as a requirement. This speeds things up substantially, but requires an Nvidia GPU with CUDA support. Additionally you will need to install CUDA and libcudnn. This can be a pain, to skip this just remove `-gpu (1.3.0)` from that line in file requirements.txt and tensorflow will run on CPU.

3. Make a directory, clone this repository from Github; cd into directory.
With virtualenvwrapper installed and configured for python3:
    ~/$: mkproject sift
    ~/source/sift$: git clone https://github.com/thesambeck/sift2
    ~/source/sift$: cd sift2

4. Install SIFT requirements (inside virtualenv) for Python 3 using PIP (you may have to type pip3 here)
   pip install -r requirements.txt

5. Edit parameters to your liking inside the relevant file, either
sift.py (for visualizations)
or
siftnonvisual.py (for rapid computation)

For nonvisual, set twitter-mode to off, or you will get errors from google-cloud (used for generating tweet text) until you have configured it.

The SIFT team has been running 999 as increment internally, so use a different number. You also can change: screen size, save/file preferences, and the range of the transform generator function (i.e. contrast) with relative ease.

5. Run SIFT (visualized or not) in Python 3:
python sift.py
OR
python sift-nonvisual.py


### This should get you started. Happy sifting!

![Image sample](https://github.com/notsambeck/sift2/most_recent.png)
