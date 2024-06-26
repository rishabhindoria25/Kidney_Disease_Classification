# Kidney_Disease_Classification

# Kidney-Disease-Classification-MLflow-DVC


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml
10. app.py

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/rishabhindoria25/Kidney_Disease_Classification.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n cnncls python=3.8 -y
```

```bash
conda activate cnncls
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtu.be/qdcHHrsXA48?si=bD5vDS60akNphkem)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/rishabhindoria25/Kidney_Disease_Classification.mlflow \
MLFLOW_TRACKING_USERNAME=rishabhindoria25 \
MLFLOW_TRACKING_PASSWORD=xxxx \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/rishabhindoria25/Kidney_Disease_Classification.mlflow

export MLFLOW_TRACKING_USERNAME=rishabhindoria25 

export MLFLOW_TRACKING_PASSWORD= xxxx

```