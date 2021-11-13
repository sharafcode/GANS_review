# GANS_review
This is a review and experimental code for different GANs models trained on MNIST dataset


## Requirements
- python 3.6+
- torch @ https://download.pytorch.org/whl/cu111/torch-1.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl
- torchvision @ https://download.pytorch.org/whl/cu111/torchvision-0.11.1%2Bcu111-cp37-cp37m-linux_x86_64.whl

## Configure the code on your environment
```
## 1. Create your python 3.+ environment

## 2. Install dependcies
>> pip install -r requirements.txt
```

### Linear GAN Batch nromalization
```
## if you want to run the code on GPU specify (--device gpu)
>> python main.py --model linear --bn batch --device cpu
```

### Linear GAN without Batch nromalization
```
>> python main.py --model linear --bn nobatch --device cpu
```

### DCGAN with Batch nromalization
```
>> python main.py --model conv --bn batch
```

### DCGAN with Spectral nromalization
```
>> python main.py --model conv --bn spectral
```
-------------

### Running on colab
- Clone the whole REPO on a colab notebook environment
- work with the notebook `main.ipynb` you will find ready-code to run different models and also a sample results saved in the notebook
