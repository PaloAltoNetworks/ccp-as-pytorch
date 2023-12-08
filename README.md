# ccp-in-pytorch

This repository implements Contrastive Credibility Propagation (CCP) in PyTorch [1].  CCP is an iterative semi-supervised learning framework that applies soft pseudolabels to unlabeled data. CCP unifies semi-supervised learning and noisy label learning for the goal of reliably outperforming a supervised baseline in any data scenario.
 
CCP has two stages. The first stage trains a neural network that predicts real-valued "q-vectors" for each unlabeled sample. Q-vectors characterize the extent to which that sample uniquely reflects each class. This stage is trained iteratively, with increasing amounts of originally-unlabeled data incorporated into learning the mapping from data to q-vectors. Each iteration reflects a full model training. In the second stage, CCP trains a classifier for the true task.  For classification model input, rather than use the original encoding for each sample (e.g., image RGB-channel flattening or text embedding), CCP uses an interim representation that is produced through the q-vector prediction task.

This repository implements CCP from the v1 paper description as a set of PyTorch classes.

## Getting started

### As a user

See the `usage-examples` directory for examples of using this codebase. 

The implementation automatically runs on a single GPU device named `cuda:0` if CUDA is available. If unavailable, it defaults to using the CPU.

### As a developer

One method:
1. `conda create -n ccp python=3.9 -y`
2. `conda activate ccp`
3. `pip install -e .[typing,test,examples]` (for package)
4. `pip install -r requirements-dev.txt` (for development environment)
5. `pre-commit install` (to get the pre-commit hooks active so you don't push unlinted/unformatted code)

And then use:

* `make lint` (`make -i lint` if you want every line to proceed)
* `make format`
* `make test`

## Authors
Xavier Mignot and Pamela Toman (Palo Alto Networks)

## References
[1] Brody Kutt, Pamela Toman, Xavier Mignot, Sujit Rokka Chhetri, Shan Huang, Nandini Ramanan, Min Du, William Hewlett. Contrastive Credibility Propagation for Reliable Semi-Supervised Learning. https://arxiv.org/abs/2211.09929v1
