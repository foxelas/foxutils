# foxutils
Foxelas utils for python for rapid development 

The latest version is in branch 'v1'.

## Contents: 
- feature_extractors
    Common models for text processing including a text_cleaner, word embedding, sentiment detection, and summarization. 
    
- utils 
    Functionalities for image and time series processing, including display plots. 
    Basic models with Keras and Torch. 
    Data generators and ARIMA models.
    Training suite with PyTorch Lightning.

- gradio

- streams 

## Installation
### Local installation:
Install locally at a virtual environment with:
```
$ python -m pip install -e .
```
To clean up the egg files:
```
$ python setup.py clean     
```

### Add package requirement: 
In the requirements.txt add the following line:
```
foxutils @ git+https://github.com/foxelas/foxutils@v1
```

