# SIFT

![travis build status](https://www.travis-ci.org/notsambeck/sift2.svg?branch=master)

### SIFT is a visual experiment. It has two parts:

1. SIFT systematically generates every possible image
2. A neural network evaluates each image to determine if it is interesting or not.
3. Candidate images are displayed and/or saved to file and/or posted to Twitter.

##### The goal is to answer a question:
##### Using brute-force combinatorics, how many images would one have to generate to find one that is indistinguishable from a photograph?

## Background

At the beginning of this project (in 2009), it seemed that the needed number of images was impossible. Or, more precisely, that it was very unlikely that any generated image would have properties we associate with real images.

Given the scale of the numbers involved, the best strategy for answering this question is to start with ‘minimal’ images - the smallest data objects that start to take on the properties of images. Fortunately, with the development of machine learning as a viable technology over the last ~10 years, scientists have started researching closely related questions. One relevant result comes from CIFAR. They have shown that scene recognition is possible with very small images, both by computers and by people. A 32x32 pixel image contains enough information to identify the context of ~90% of cases. This applies to both humans and computers. As an extension of this result, it should be possible to classify images as 'real' or 'simulated' with relative ease. The goal is to find simulated images that `function` as real images.

The packaged neural net has been trained on hundreds of thousands of generated images and real images from CIFAR and ImageNet datasets. 

As of November 2017, over 10^10 images have been generated and analyzed. None have passed the ultimate human test for passing as photographs, but some have been interesting.


## Installation
### Version 0.2 is stable and configured to run out-of-the-box on Linux 16.04 / Python3.

(If you are using older Linux or most other OS, you may have to specify Python3 / PIP3.)

1. Install python3, git (and optionally virtualenv + virtualenvwrapper) globally

2. Tensorflow is a requirement (runs neural net). requirements.txt lists the vanilla version, but installing with GPU support and/or building from source will provide significant performance gains. 

3. Clone repository from Github. With virtualenvwrapper installed (_configured for python3_) the following works:
```bash
~/$: mkproject sift
~/sift$: git clone https://github.com/thesambeck/sift2 .
```

4. Install SIFT requirements (inside virtualenv) for Python 3 using PIP (you may have to type pip3 here). If an error about Cython appears, you may be able to simply `pip install -I Cython==0.23`, then attempt to install the requirements again.
```bash
~/sift$: pip install -r requirements.txt
```

5. Decompress neural net weights .h5 file (unless you want to train up your own network; if you don't need the 68MB net_{current_version}.h5 file you, can clone the no_net_files branch). This file is compressed because otherwise it exceeds the github 100MB file size limit.

```bash
~/sift$: cd net
~/sift/net$: gzip -d keras_net_v0_2017aug7.h5.gz
```

6. Edit parameters inside the relevant file: `sift.py` (for visualized version), `siftnonvisual.py` (for rapid computation), or `dataset.py` (lower level tools & image generator).

For siftnonvisual, **keep twitter-mode off**, or you may get errors from google-cloud (their API is used for generating tweet text and object recognition) until you have configured it.

The SIFT team has been running 999 and 11999 as increment internally; use a different seed or, better yet, change the parameter for `quantization` in the module `dataset`, which will put your instance on a divergent path. This ensures you are not duplicating existing images. One can also adjust: screen size (for `sift.py`), save/file/output preferences, and the range of the transform generator function (i.e. contrast) with relative ease.

5. To run SIFT (visualized or not) in Python 3:
```~/sift$: python sift.py```
OR
```~/sift$: python siftnonvisual.py```

## More info
www.notsambeck.com

## Happy sifting!
![Most recent SIFT image sample](https://github.com/notsambeck/sift2/blob/master/most_recent.png)
