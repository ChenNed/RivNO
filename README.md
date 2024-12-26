# RivNO
Graph-enhanced Neural Operator for Missing Velocities Infilling in River Surface Velocimetry


## Environment Setup
Please install the following libraries:
```
Pytorch
PyTorch Geometric
tqdm
yaml
numpy
pickle
sklearn
```

## Folders
```Shell
├── configs
    ├──operator
      ├──no.yaml # Parameter configuration
├── data
├── datasets
        ├── data_utils.py
        ├── dataset_create.py
        ├── graph_construction.py
        ├── inferdata_create.py
        ├── masking_strategies.py
        ├── scaling.py
├── models
        ├── integral.py
        ├── layers.py
        ├── pino.py
├── train_utils
        ├── adam.py
        ├── inference.py
        ├── loss.py
        ├── metrics.py
        ├── test.py
        ├── train.py
```

## Training
Model-specific parameters can be found in the no.yaml file.
```
python train.py # You may need to adjust the paths.
```
## Testing
Testing with specific checkpoints.
```
python test.py # You may need to adjust the paths and the specific checkpoint you want to use.
```

## Inference
Inference for unseen samples.
```
python inference.py # You may need to adjust the paths.
```



## Acknowledgement
Parts of the code are adapted from the following repository. We thank the authors for their great contribution to the community:
- [NeuralOperator]: Learning in Infinite Dimensions]([https://github.com/princeton-vl/RAFT/tree/master](https://github.com/neuraloperator/neuraloperator)
