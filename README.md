# PyTorch-Adversarial-PGD-Training

PyTorch implementation of [Adversarial PGD Training](https://arxiv.org/abs/1706.06083).

## Summary

This work is based on [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083). **CIFAR-10** is used as the dataset. Adversarial PGD training starts with pretrained model from [PyTorchCV](https://pypi.org/project/pytorchcv/). You should be able to change the code into different datasets such as [ImageNet](http://www.image-net.org), [CIFAR-10/CIFAR-100](https://www.cs.toronto.edu/%7Ekriz/cifar.html), [SVHN](http://ufldl.stanford.edu/housenumbers/) or different models (see [model list](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py)) for adversarial training.

## Requirements
```bash
pip3 install pytorchcv
```

## Train

### Run
```bash
python3 train.py
```

### Default Settings
* batch size = 128
* SGD optimizer with learning rate = 0.1
* wrn16 for adversarial training
* PGD
    * distance measurement: L-infinity
    * epsilon (maximum perturbation) = 8
    * alpha (step size) = 0.8
    * num_iter (number of step) = 20

### Performance

train accuracy: white box PGD attack accuracy evaluated on the training set (50000 images)
test accuracy: white box PGD attack accuracy evaluated on the testing set (10000 images)

![](https://i.imgur.com/YFht8Rn.png)

best performance: epoch 7, test accuracy = 48.5%

## Test

### Download

```bash
pip3 install gdown
bash setup.sh
```
or download data below
* [wrn16 best state dict](https://drive.google.com/file/d/1QsXu3FpU5N4pXN4Vs5r2QR6Y-Y79hdpK/view?usp=sharing)
* [sample benign images](https://drive.google.com/file/d/1F0Jye2aOHAtSoknMV-ElPVC7UtqMTKAs/view?usp=sharing)
* [sample adversarial images](https://drive.google.com/file/d/1Y-3PPHZuOcATU-SSFCdBn1uyRLfiV9AD/view?usp=sharing)

### Run

```bash
python3 test.py
```

## Modification

This section points out the part you may need to modify if you would like to change the dataset or the model for adversarial training.

### Dataset (from data.py and train.py)
```python
from torchvision.datasets import CIFAR10

cifar_10_mean = (0.491, 0.482, 0.447)
cifar_10_std = (0.202, 0.199, 0.201)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
```

### Model ([model list](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py))
```bash
python3 train.py --model <model name from model list>
```

## Reference

* [PyTorchCV](https://pypi.org/project/pytorchcv/)
* [A Great Tutorial About Adversarial Attack](https://adversarial-ml-tutorial.org)
