SPADE-adv\_train
===============================

The adversarial training application of SPADE. For MNIST, we adapt the Tensorflow code from [MadryLab's repo](https://github.com/MadryLab/mnist_challenge/). As for CIFAR10, we adapt the PyTorch code from [locuslab's repo](https://github.com/locuslab/fast_adversarial/tree/master/CIFAR10) to speedup the adversarial training from 3 days to 2 hours on a single GPU.


Requirements
------------
* tensorflow <= 1.15 (Only required for MNIST)
* googledrivedownloader (Only required if you fetch the pre-trained models for MNIST)
* [Requirements in locuslab/fast_adversarial](https://github.com/locuslab/fast_adversarial/tree/master/CIFAR10) (Only required for CIFAR10)
* Julia (Only required for computing node SPADE score on CIFAR10)
* hnswlib (Only required for computing node SPADE score on CIFAR10)


Usage
-----

**MNIST Usage**

1. `cd mnist/`

2. `python fetch_model.py --method spade` (Only required if you fetch the pre-trained model)

3. `python train.py --device 0 --method spade` (Only required if you train the model from scratch)

4. `python eval.py --device 0 --method spade`


**CIFAR10 Usage**

1. `cd cifar10/`

2. `python train.py --method spade` (Only required if you train the model from scratch)

3. `python eval.py --method spade`
