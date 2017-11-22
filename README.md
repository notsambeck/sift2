# SIFT

![travis build status]https://www.travis-ci.org/notsambeck/sift2.svg?branch=master

### SIFT is a visual experiment. It has two parts:

1. SIFT systematically generates every possible image
2. A neural network evaluates each image to determine if it is interesting or not. Candidate images are displayed and/or saved to file.

### The goal is to answer a question:
### Using brute-force combinatorics, how many images would one have to generate to find one that is indistinguishable from a photograph?

At the beginning of this project (in 2009), it seemed that the needed number of images was impossible. Or, more precisely, that it was ridiculously unlikely that any generated image would have properties we associate with real images.

Given the scale of the numbers involved, the best strategy for answering this question is to start with ‘minimal’ images - the smallest data objects that start to take on the properties of images. Fortunately, with the development of machine learning as a viable technology over the last ~10 years, scientists have started researching closely related questions. One relevant result comes from CIFAR, an academic project that studies computer vision. They have shown that scene recognition is possible with very small images, both by computers and by people. A 32x32 pixel image contains enough information to identify the context of ~90% of cases. This applies to both humans and computers.


SIFT is built with many pre-existing tools, which can be found in requirements.txt. It is not dependent on CUDA, but it will benefit from CUDA / GPU acceleration.

### INSTALLATION for Linux 16
(If you are using older Linux or most other OS, you will probably have to specify Python3 and PIP3.)

1. Install python3, git (and optionally virtualenv + virtualenvwrapper) globally

2. Requirements.txt lists tensorflow-gpu as a requirement. This requires an Nvidia GPU with CUDA support, plus installing CUDA and libcudnn. **To skip this just remove `-gpu (1.3.0)` from that line in file requirements.txt**; tensorflow will run on CPU. Good for a trial run.

3. Clone this repository from Github
With virtualenvwrapper installed and configured for python3:
```bash
~/$: mkproject sift
and/or
~/$: workon sift
~/source/sift$: git clone https://github.com/thesambeck/sift2 .
```

4. Install SIFT requirements (inside virtualenv) for Python 3 using PIP (you may have to type pip3 here)
```bash
pip install -r requirements.txt
```

5. Decompress neural net weights .h5 file (unless you want to train up your own network; if you don't need the 68MB net_{current_version}.h5 file you, can clone the no_net_files branch). This file is compressed because otherwise it exceeds the github 100MB file size limit.
```bash
~/source/sift$: cd net
~/source/sift/net$: gzip -d keras_net_v0_2017aug7.h5.gz
```

6. Edit parameters to your liking inside the relevant file: `sift.py` (for visualizations), `siftnonvisual.py` (for rapid computation), or `dataset.py` (lower level tools & image generator).

For siftnonvisual, **turn twitter-mode off**, or you will get errors from google-cloud (their API is used for generating tweet text and object recognition) until you have configured it.

The SIFT team has been running 999 as increment internally; use a different seed to avoid duplicating work! You also can change: screen size for sift.py, save/file/output preferences, and the range of the transform generator function (i.e. contrast) with relative ease.

5. Run SIFT (visualized or not) in Python 3:
```python sift.py```
OR
```python siftnonvisual.py```


### This should get you started. Happy sifting!

![Most recent SIFT image sample](https://github.com/notsambeck/sift2/blob/master/most_recent.png)
