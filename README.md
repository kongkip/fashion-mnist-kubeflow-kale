# fashion-mnist-kubeflow-kale

This repo contains the code for training and fashion mnist dataset utilizing 
kubeflow pipelines with kubeflow kale.

kubeflow is an MLOPS tool build for orchestrating machine learning pipelines 
and allows for reproducibility. The 
repo uses kubeflow-kale which is a simplified sdk for defining the pipelines.


## Usage
```
python3 -m kale main.py --kfp
```