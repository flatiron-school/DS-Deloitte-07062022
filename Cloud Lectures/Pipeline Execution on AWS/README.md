**Note:** *After having covered the material in the [Pipeline Creation](https://github.com/flatiron-school/DS-Deloitte-07062022-Architecting-Pipelines-with-AWS/blob/main/Pipeline%20Creation.ipynb) and [Pipeline Execution](https://github.com/flatiron-school/DS-Deloitte-07062022-Architecting-Pipelines-with-AWS/blob/main/Pipeline%20Execution.ipynb) notebooks, the content in this notebook is intended to demonstrate an alternative workflow to that of the batch transformation pipeline -- the creation of a real-time inference endpoint!*

![](images/aws-model-inference-options-2.png)

## Contents

1. [Introduction](#Introduction)
2. [Setup](#Setup)
3. [Training the XGBoost model](#Training-the-XGBoost-model)
4. [Deploying the XGBoost endpoint](#Deploying-the-XGBoost-endpoint)
5. [Explaining Model Predictions](#Explain-the-model's-predictions-on-each-data-point)
6. [Delete the Inference Endpoint](#Delete-Endpoint)

[Source](https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_inferenece_script_mode.html)

## Introduction
    
This notebook shows how you can configure the SageMaker XGBoost model server by defining the following three functions in the Python source file you pass to the XGBoost constructor in the SageMaker Python SDK:
- `input_fn`: Takes request data and deserializes the data into an object for prediction,
- `predict_fn`: Takes the deserialized request object and performs inference against the loaded model, and
- `output_fn`: Takes the result of prediction and serializes this according to the response content type.
We will write a customized inference script that is designed to illustrate how [SHAP](https://github.com/slundberg/shap) values enable the interpretion of XGBoost models.

We use the [Abalone data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html), originally from the [UCI data repository](https://archive.ics.uci.edu/ml/datasets/abalone). More details about the original dataset can be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names).  In this libsvm converted version, the nominal feature (Male/Female/Infant) has been converted into a real valued feature as required by XGBoost. Age of abalone is to be predicted from eight physical measurements.

This notebook uses the Abalone dataset to deploy a model server that returns SHAP values, which enable us to create model explanation such as the following plots that show each features contributing to push the model output from the base value.

<table><tr>
    <td> <img src="images/output_8_0.png"/> </td>
    <td> <img src="images/output_9_0.png"/> </td>
</tr></table>

## Setup
    
This notebook was tested in Amazon SageMaker Studio on a `ml.t3.medium` instance.

Let's start by specifying:

1. The S3 bucket and prefix that you want to use for training and model data. This should be within the same region as the Notebook Instance, training, and hosting.
2. The IAM role arn used to give training and hosting access to your data. See the documentation for how to create these. Note, if more than one role is required for notebook instances, training, and/or hosting, please replace the boto regex with a the appropriate full IAM role arn string(s).


```python
%%time

import io
import os
import boto3
import sagemaker
import time

role = sagemaker.get_execution_role()
region = boto3.Session().region_name

# S3 bucket for saving code and model artifacts.
# Feel free to specify a different bucket here if you wish.
bucket = sagemaker.Session().default_bucket()
prefix = "sagemaker/DEMO-xgboost-inference-script-mode"
```

    CPU times: user 971 ms, sys: 172 ms, total: 1.14 s
    Wall time: 18.6 s


### Fetching the dataset

The following methods download the Abalone dataset and upload files to S3:


```python
%%time
s3 = boto3.client("s3")
# Load the dataset
FILE_DATA = "abalone"
s3.download_file(
    "sagemaker-sample-files", f"datasets/tabular/uci_abalone/abalone.libsvm", FILE_DATA
)
sagemaker.Session().upload_data(FILE_DATA, bucket=bucket, key_prefix=prefix + "/train")
```

    CPU times: user 151 ms, sys: 15.6 ms, total: 167 ms
    Wall time: 976 ms





    's3://sagemaker-us-west-1-167762637358/sagemaker/DEMO-xgboost-inference-script-mode/train/abalone'



## Training the XGBoost model
    
SageMaker can now run an XGboost script using the XGBoost estimator. A typical training script loads data from the input channels, configures training with hyperparameters, trains a model, and saves a model to `model_dir` so that it can be hosted later. In this notebook, we use the same training script [abalone.py](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/abalone.py) from [Regression with Amazon SageMaker XGBoost algorithm](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone_dist_script_mode.ipynb). Refer to [Regression with Amazon SageMaker XGBoost algorithm](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone_dist_script_mode.ipynb) for details on the training script.

After setting training parameters, we kick off training, and poll for status until training is completed, which in this example, takes between few minutes.

To run our training script on SageMaker, we construct a `sagemaker.xgboost.estimator.XGBoost` estimator, which accepts several constructor arguments:

* __entry_point__: The path to the Python script SageMaker runs for training and prediction.
* __role__: Role ARN
* __framework_version__: SageMaker XGBoost version you want to use for executing your model training code, e.g., `1.0-1`, `1.2-2`, `1.3-1`, or `1.5-1`.
* __train_instance_type__ *(optional)*: The type of SageMaker instances for training. __Note__: Because Scikit-learn does not natively support GPU training, Sagemaker Scikit-learn does not currently support training on GPU instance types.
* __sagemaker_session__ *(optional)*: The session used to train on Sagemaker.
* __hyperparameters__ *(optional)*: A dictionary passed to the train function as hyperparameters.


```python
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost

job_name = "DEMO-xgboost-inference-script-mode-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print("Training job", job_name)

hyperparameters = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "objective": "reg:squarederror",
    "num_round": "50",
    "verbosity": "2",
}

instance_type = "ml.c5.xlarge"

xgb_script_mode_estimator = XGBoost(
    entry_point="abalone.py",
    hyperparameters=hyperparameters,
    role=role,
    instance_count=1,
    instance_type=instance_type,
    framework_version="1.5-1",
    output_path="s3://{}/{}/{}/output".format(bucket, prefix, job_name),
)

content_type = "text/libsvm"
train_input = TrainingInput(
    "s3://{}/{}/{}/".format(bucket, prefix, "train"), content_type=content_type
)
```

    Training job DEMO-xgboost-inference-script-mode-2022-08-18-20-51-25


### Train XGBoost Estimator on Abalone Data 
    
Training is as simple as calling `fit` on the Estimator. This will start a SageMaker Training job that will download the data, invoke the entry point code (in the provided script file), and save any model artifacts that the script creates. In this case, the script requires a `train` and a `validation` channel. Since we only created a `train` channel, we re-use it for validation:


```python
xgb_script_mode_estimator.fit({"train": train_input, "validation": train_input}, job_name=job_name)
```

    2022-08-18 20:53:40 Starting - Starting the training job...
    2022-08-18 20:53:55 Starting - Preparing the instances for trainingProfilerReport-1660856020: InProgress
    ......
    2022-08-18 20:55:09 Downloading - Downloading input data...
    2022-08-18 20:55:38 Training - Downloading the training image.....[34m[2022-08-18 20:56:18.701 ip-10-0-180-2.us-west-1.compute.internal:1 INFO utils.py:27] RULE_JOB_STOP_SIGNAL_FILENAME: None[0m
    [34m[2022-08-18:20:56:18:INFO] Imported framework sagemaker_xgboost_container.training[0m
    [34m[2022-08-18:20:56:18:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m[2022-08-18:20:56:18:INFO] Invoking user training script.[0m
    [34m[2022-08-18:20:56:19:INFO] Module abalone does not provide a setup.py. [0m
    [34mGenerating setup.py[0m
    [34m[2022-08-18:20:56:19:INFO] Generating setup.cfg[0m
    [34m[2022-08-18:20:56:19:INFO] Generating MANIFEST.in[0m
    [34m[2022-08-18:20:56:19:INFO] Installing module with the following command:[0m
    [34m/miniconda3/bin/python3 -m pip install . [0m
    [34mProcessing /opt/ml/code
      Preparing metadata (setup.py): started[0m
    [34m  Preparing metadata (setup.py): finished with status 'done'[0m
    [34mBuilding wheels for collected packages: abalone
      Building wheel for abalone (setup.py): started[0m
    [34m  Building wheel for abalone (setup.py): finished with status 'done'
      Created wheel for abalone: filename=abalone-1.0.0-py2.py3-none-any.whl size=5714 sha256=0dfd94cd1fdeb5cbd3d08d6b34bc4a403e39e793e200bc99e9a21fe8eeed610c
      Stored in directory: /home/model-server/tmp/pip-ephem-wheel-cache-mwg1ueyv/wheels/f3/75/57/158162e9eab7af12b5c338c279b3a81f103b89d74eeb911c00[0m
    [34mSuccessfully built abalone[0m
    [34mInstalling collected packages: abalone[0m
    [34mSuccessfully installed abalone-1.0.0[0m
    [34mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m
    [34m[notice] A new release of pip available: 22.1.2 -> 22.2.2[0m
    [34m[notice] To update, run: pip install --upgrade pip[0m
    [34m[2022-08-18:20:56:23:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m[2022-08-18:20:56:23:INFO] Invoking user script[0m
    [34mTraining Env:[0m
    [34m{
        "additional_framework_parameters": {},
        "channel_input_dirs": {
            "train": "/opt/ml/input/data/train",
            "validation": "/opt/ml/input/data/validation"
        },
        "current_host": "algo-1",
        "framework_module": "sagemaker_xgboost_container.training:main",
        "hosts": [
            "algo-1"
        ],
        "hyperparameters": {
            "eta": "0.2",
            "gamma": "4",
            "max_depth": "5",
            "min_child_weight": "6",
            "num_round": "50",
            "objective": "reg:squarederror",
            "subsample": "0.7",
            "verbosity": "2"
        },
        "input_config_dir": "/opt/ml/input/config",
        "input_data_config": {
            "train": {
                "ContentType": "text/libsvm",
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            },
            "validation": {
                "ContentType": "text/libsvm",
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            }
        },
        "input_dir": "/opt/ml/input",
        "is_master": true,
        "job_name": "DEMO-xgboost-inference-script-mode-2022-08-18-20-51-25",
        "log_level": 20,
        "master_hostname": "algo-1",
        "model_dir": "/opt/ml/model",
        "module_dir": "s3://sagemaker-us-west-1-167762637358/DEMO-xgboost-inference-script-mode-2022-08-18-20-51-25/source/sourcedir.tar.gz",
        "module_name": "abalone",
        "network_interface_name": "eth0",
        "num_cpus": 4,
        "num_gpus": 0,
        "output_data_dir": "/opt/ml/output/data",
        "output_dir": "/opt/ml/output",
        "output_intermediate_dir": "/opt/ml/output/intermediate",
        "resource_config": {
            "current_host": "algo-1",
            "current_instance_type": "ml.c5.xlarge",
            "current_group_name": "homogeneousCluster",
            "hosts": [
                "algo-1"
            ],
            "instance_groups": [
                {
                    "instance_group_name": "homogeneousCluster",
                    "instance_type": "ml.c5.xlarge",
                    "hosts": [
                        "algo-1"
                    ]
                }
            ],
            "network_interface_name": "eth0"
        },
        "user_entry_point": "abalone.py"[0m
    [34m}[0m
    [34mEnvironment variables:[0m
    [34mSM_HOSTS=["algo-1"][0m
    [34mSM_NETWORK_INTERFACE_NAME=eth0[0m
    [34mSM_HPS={"eta":"0.2","gamma":"4","max_depth":"5","min_child_weight":"6","num_round":"50","objective":"reg:squarederror","subsample":"0.7","verbosity":"2"}[0m
    [34mSM_USER_ENTRY_POINT=abalone.py[0m
    [34mSM_FRAMEWORK_PARAMS={}[0m
    [34mSM_RESOURCE_CONFIG={"current_group_name":"homogeneousCluster","current_host":"algo-1","current_instance_type":"ml.c5.xlarge","hosts":["algo-1"],"instance_groups":[{"hosts":["algo-1"],"instance_group_name":"homogeneousCluster","instance_type":"ml.c5.xlarge"}],"network_interface_name":"eth0"}[0m
    [34mSM_INPUT_DATA_CONFIG={"train":{"ContentType":"text/libsvm","RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"},"validation":{"ContentType":"text/libsvm","RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}[0m
    [34mSM_OUTPUT_DATA_DIR=/opt/ml/output/data[0m
    [34mSM_CHANNELS=["train","validation"][0m
    [34mSM_CURRENT_HOST=algo-1[0m
    [34mSM_MODULE_NAME=abalone[0m
    [34mSM_LOG_LEVEL=20[0m
    [34mSM_FRAMEWORK_MODULE=sagemaker_xgboost_container.training:main[0m
    [34mSM_INPUT_DIR=/opt/ml/input[0m
    [34mSM_INPUT_CONFIG_DIR=/opt/ml/input/config[0m
    [34mSM_OUTPUT_DIR=/opt/ml/output[0m
    [34mSM_NUM_CPUS=4[0m
    [34mSM_NUM_GPUS=0[0m
    [34mSM_MODEL_DIR=/opt/ml/model[0m
    [34mSM_MODULE_DIR=s3://sagemaker-us-west-1-167762637358/DEMO-xgboost-inference-script-mode-2022-08-18-20-51-25/source/sourcedir.tar.gz[0m
    [34mSM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"train":"/opt/ml/input/data/train","validation":"/opt/ml/input/data/validation"},"current_host":"algo-1","framework_module":"sagemaker_xgboost_container.training:main","hosts":["algo-1"],"hyperparameters":{"eta":"0.2","gamma":"4","max_depth":"5","min_child_weight":"6","num_round":"50","objective":"reg:squarederror","subsample":"0.7","verbosity":"2"},"input_config_dir":"/opt/ml/input/config","input_data_config":{"train":{"ContentType":"text/libsvm","RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"},"validation":{"ContentType":"text/libsvm","RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","is_master":true,"job_name":"DEMO-xgboost-inference-script-mode-2022-08-18-20-51-25","log_level":20,"master_hostname":"algo-1","model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-us-west-1-167762637358/DEMO-xgboost-inference-script-mode-2022-08-18-20-51-25/source/sourcedir.tar.gz","module_name":"abalone","network_interface_name":"eth0","num_cpus":4,"num_gpus":0,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_group_name":"homogeneousCluster","current_host":"algo-1","current_instance_type":"ml.c5.xlarge","hosts":["algo-1"],"instance_groups":[{"hosts":["algo-1"],"instance_group_name":"homogeneousCluster","instance_type":"ml.c5.xlarge"}],"network_interface_name":"eth0"},"user_entry_point":"abalone.py"}[0m
    [34mSM_USER_ARGS=["--eta","0.2","--gamma","4","--max_depth","5","--min_child_weight","6","--num_round","50","--objective","reg:squarederror","--subsample","0.7","--verbosity","2"][0m
    [34mSM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate[0m
    [34mSM_CHANNEL_TRAIN=/opt/ml/input/data/train[0m
    [34mSM_CHANNEL_VALIDATION=/opt/ml/input/data/validation[0m
    [34mSM_HP_ETA=0.2[0m
    [34mSM_HP_GAMMA=4[0m
    [34mSM_HP_MAX_DEPTH=5[0m
    [34mSM_HP_MIN_CHILD_WEIGHT=6[0m
    [34mSM_HP_NUM_ROUND=50[0m
    [34mSM_HP_OBJECTIVE=reg:squarederror[0m
    [34mSM_HP_SUBSAMPLE=0.7[0m
    [34mSM_HP_VERBOSITY=2[0m
    [34mPYTHONPATH=/miniconda3/bin:/:/miniconda3/lib/python/site-packages/xgboost/dmlc-core/tracker:/miniconda3/lib/python38.zip:/miniconda3/lib/python3.8:/miniconda3/lib/python3.8/lib-dynload:/miniconda3/lib/python3.8/site-packages[0m
    [34mInvoking script with the following command:[0m
    [34m/miniconda3/bin/python3 -m abalone --eta 0.2 --gamma 4 --max_depth 5 --min_child_weight 6 --num_round 50 --objective reg:squarederror --subsample 0.7 --verbosity 2[0m
    [34m[20:56:24] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 40 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[0]#011train-rmse:8.09085#011validation-rmse:8.09085[0m
    [34m[20:56:24] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 38 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[1]#011train-rmse:6.61129#011validation-rmse:6.61129[0m
    [34m[20:56:24] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 40 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[2]#011train-rmse:5.44558#011validation-rmse:5.44558[0m
    [34m[20:56:24] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 38 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[3]#011train-rmse:4.54894#011validation-rmse:4.54894[0m
    [34m[20:56:24] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 48 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[4]#011train-rmse:3.85379#011validation-rmse:3.85379[0m
    [34m[20:56:24] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 52 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[5]#011train-rmse:3.32450#011validation-rmse:3.32450[0m
    [34m[20:56:24] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 44 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[6]#011train-rmse:2.92907#011validation-rmse:2.92907[0m
    [34m[20:56:24] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 44 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[7]#011train-rmse:2.64925#011validation-rmse:2.64925[0m
    [34m[20:56:24] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 52 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[8]#011train-rmse:2.43828#011validation-rmse:2.43828[0m
    [34m[20:56:24] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 48 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[9]#011train-rmse:2.28504#011validation-rmse:2.28504[0m
    [34m[20:56:24] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 52 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[10]#011train-rmse:2.17756#011validation-rmse:2.17756[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 42 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[11]#011train-rmse:2.10257#011validation-rmse:2.10257[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 46 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[12]#011train-rmse:2.04681#011validation-rmse:2.04681[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 42 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[13]#011train-rmse:2.00737#011validation-rmse:2.00737[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 32 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[14]#011train-rmse:1.97778#011validation-rmse:1.97778[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 44 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[15]#011train-rmse:1.95060#011validation-rmse:1.95060[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 42 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[16]#011train-rmse:1.93036#011validation-rmse:1.93036[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 26 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[17]#011train-rmse:1.91997#011validation-rmse:1.91997[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 44 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[18]#011train-rmse:1.90255#011validation-rmse:1.90255[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 56 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[19]#011train-rmse:1.88461#011validation-rmse:1.88461[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 32 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[20]#011train-rmse:1.87660#011validation-rmse:1.87660[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 40 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[21]#011train-rmse:1.86282#011validation-rmse:1.86282[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 30 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[22]#011train-rmse:1.85499#011validation-rmse:1.85499[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 20 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[23]#011train-rmse:1.84877#011validation-rmse:1.84877[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 38 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[24]#011train-rmse:1.84014#011validation-rmse:1.84014[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 18 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[25]#011train-rmse:1.83703#011validation-rmse:1.83703[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 42 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[26]#011train-rmse:1.82825#011validation-rmse:1.82825[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 14 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[27]#011train-rmse:1.82615#011validation-rmse:1.82615[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 52 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[28]#011train-rmse:1.81786#011validation-rmse:1.81786[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 30 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[29]#011train-rmse:1.81118#011validation-rmse:1.81118[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 42 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[30]#011train-rmse:1.80298#011validation-rmse:1.80298[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 32 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[31]#011train-rmse:1.79704#011validation-rmse:1.79704[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 38 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[32]#011train-rmse:1.78973#011validation-rmse:1.78973[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 40 extra nodes, 8 pruned nodes, max_depth=5[0m
    [34m[33]#011train-rmse:1.78096#011validation-rmse:1.78096[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 12 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[34]#011train-rmse:1.77939#011validation-rmse:1.77939[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 16 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[35]#011train-rmse:1.77711#011validation-rmse:1.77711[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 28 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[36]#011train-rmse:1.77266#011validation-rmse:1.77266[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 26 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[37]#011train-rmse:1.76877#011validation-rmse:1.76877[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 26 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[38]#011train-rmse:1.76343#011validation-rmse:1.76343[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 36 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[39]#011train-rmse:1.75774#011validation-rmse:1.75774[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 38 extra nodes, 4 pruned nodes, max_depth=5[0m
    [34m[40]#011train-rmse:1.75110#011validation-rmse:1.75110[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 22 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[41]#011train-rmse:1.74668#011validation-rmse:1.74668[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 24 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[42]#011train-rmse:1.74404#011validation-rmse:1.74404[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 26 extra nodes, 6 pruned nodes, max_depth=5[0m
    [34m[43]#011train-rmse:1.74232#011validation-rmse:1.74232[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 24 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[44]#011train-rmse:1.73694#011validation-rmse:1.73694[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 20 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[45]#011train-rmse:1.73464#011validation-rmse:1.73464[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 34 extra nodes, 2 pruned nodes, max_depth=5[0m
    [34m[46]#011train-rmse:1.72677#011validation-rmse:1.72677[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 24 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[47]#011train-rmse:1.72361#011validation-rmse:1.72361[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 30 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[48]#011train-rmse:1.71716#011validation-rmse:1.71716[0m
    [34m[20:56:25] INFO: ../src/tree/updater_prune.cc:101: tree pruning end, 42 extra nodes, 0 pruned nodes, max_depth=5[0m
    [34m[49]#011train-rmse:1.70623#011validation-rmse:1.70623[0m
    
    2022-08-18 20:56:38 Uploading - Uploading generated training model
    2022-08-18 20:56:58 Completed - Training job completed
    Training seconds: 97
    Billable seconds: 97


After training, we can host the newly created model in SageMaker, and create an Amazon SageMaker endpoint – a hosted and managed prediction service that we can use to perform inference. If you call `deploy` after you call `fit` on an XGBoost estimator, it will create a SageMaker endpoint using the training script (i.e., `entry_point`). You can also optionally specify other functions to customize the behavior of deserialization of the input request (`input_fn()`), serialization of the predictions (`output_fn()`), and how predictions are made (`predict_fn()`). If any of these functions are not specified, the endpoint will use the default functions in the SageMaker XGBoost container. See the [SageMaker Python SDK documentation](https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html#sagemaker-xgboost-model-server) for details.
    
In this notebook, we will run a separate inference script and customize the endpoint to return [SHAP](https://github.com/slundberg/shap) values in addition to predictions. The inference script that we will run in this notebook is provided as the accompanying file (`inference.py` | [Link](https://github.com/flatiron-school/DS-Deloitte-07062022-Architecting-Pipelines-with-AWS/blob/main/helpers/inference.py)) and also shown below:

```
import json
import os
import pickle as pkl

import numpy as np

import sagemaker_xgboost_container.encoder as xgb_encoders


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    model_file = "xgboost-model"
    booster = pkl.load(open(os.path.join(model_dir, model_file), "rb"))
    return booster


def input_fn(request_body, request_content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.

    Return a DMatrix (an object that can be passed to predict_fn).
    """
    if request_content_type == "text/libsvm":
        return xgb_encoders.libsvm_to_dmatrix(request_body)
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )


def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.

    Return a two-dimensional NumPy array where the first columns are predictions
    and the remaining columns are the feature contributions (SHAP values) for that prediction.
    """
    prediction = model.predict(input_data)
    feature_contribs = model.predict(input_data, pred_contribs=True, validate_features=False)
    output = np.hstack((prediction[:, np.newaxis], feature_contribs))
    return output


def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    if content_type == "text/csv":
        return ','.join(str(x) for x in predictions[0])
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))
```

## Deploying the XGBoost endpoint

### Deploy to an endpoint
    
Since the inference script is separate from the training script, here we use `XGBoostModel` to create a model from s3 artifacts and specify `inference.py` as the `entry_point`:


```python
from sagemaker.xgboost.model import XGBoostModel

model_data = xgb_script_mode_estimator.model_data
print(model_data)

xgb_inference_model = XGBoostModel(
    model_data=model_data,
    role=role,
    entry_point="inference.py",
    framework_version="1.5-1",
)
```

    s3://sagemaker-us-west-1-167762637358/sagemaker/DEMO-xgboost-inference-script-mode/DEMO-xgboost-inference-script-mode-2022-08-18-20-51-25/output/DEMO-xgboost-inference-script-mode-2022-08-18-20-51-25/output/model.tar.gz



```python
predictor = xgb_inference_model.deploy(
    initial_instance_count=1,
    instance_type="ml.c5.xlarge",
)
```

    -----!

## Explain the model's predictions on each data point


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_feature_contributions(prediction):

    attribute_names = [
        "Sex",  # nominal / -- / M, F, and I (infant)
        "Length",  # continuous / mm / Longest shell measurement
        "Diameter",  # continuous / mm / perpendicular to length
        "Height",  # continuous / mm / with meat in shell
        "Whole weight",  # continuous / grams / whole abalone
        "Shucked weight",  # continuous / grams / weight of meat
        "Viscera weight",  # continuous / grams / gut weight (after bleeding)
        "Shell weight",  # continuous / grams / after being dried
    ]

    prediction, _, *shap_values, bias = prediction

    if len(shap_values) != len(attribute_names):
        raise ValueError("Length mismatch between shap values and attribute names.")

    df = pd.DataFrame(data=[shap_values], index=["SHAP"], columns=attribute_names).T
    df.sort_values(by="SHAP", inplace=True)

    df["bar_start"] = bias + df.SHAP.cumsum().shift().fillna(0.0)
    df["bar_end"] = df.bar_start + df.SHAP
    df[["bar_start", "bar_end"]] = np.sort(df[["bar_start", "bar_end"]].values)
    df["hue"] = df.SHAP.apply(lambda x: 0 if x > 0 else 1)

    sns.set(style="white")

    ax1 = sns.barplot(x=df.bar_end, y=df.index, data=df, orient="h", palette="vlag")
    for idx, patch in enumerate(ax1.patches):
        x_val = patch.get_x() + patch.get_width() + 0.8
        y_val = patch.get_y() + patch.get_height() / 2
        shap_value = df.SHAP.values[idx]
        value = "{0}{1:.2f}".format("+" if shap_value > 0 else "-", shap_value)
        ax1.annotate(value, (x_val, y_val), ha="right", va="center")

    ax2 = sns.barplot(x=df.bar_start, y=df.index, data=df, orient="h", color="#FFFFFF")
    ax2.set_xlim(
        df[["bar_start", "bar_end"]].values.min() - 1, df[["bar_start", "bar_end"]].values.max() + 1
    )
    ax2.axvline(x=bias, color="#000000", alpha=0.2, linestyle="--", linewidth=1)
    ax2.set_title("base value: {0:.1f}  →  model output: {1:.1f}".format(bias, prediction))
    ax2.set_xlabel("Abalone age")

    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()


def predict_and_plot(predictor, libsvm_str):
    label, *features = libsvm_str.strip().split()
    predictions = predictor.predict(" ".join(["-99"] + features))  # use dummy label -99
    np_array = np.array([float(x) for x in predictions[0]])
    plot_feature_contributions(np_array)
```

    Matplotlib is building the font cache; this may take a moment.


The below figure shows features each contributing to push the model output from the base value (9.9 rings) to the model output (6.9 rings). The primary indicator for a young abalone according to the model is low shell weight, which decreases the prediction by 3.0 rings from the base value of 9.9 rings. Whole weight and shucked weight are also powerful indicators. The whole weight pushes the prediction lower by 0.84 rings, while shucked weight pushes the prediction higher by 1.6 rings:


```python
a_young_abalone = "6 1:3 2:0.37 3:0.29 4:0.095 5:0.249 6:0.1045 7:0.058 8:0.067"
predict_and_plot(predictor, a_young_abalone)
```


    
![png](images/output_8_0.png)
    


The second example shows feature contributions for another sample, an old abalone. We again see that the primary indicator for the age of abalone according to the model is shell weight, which increases the model prediction by 2.36 rings. Whole weight and shucked weight also contribute significantly, and they both push the model's prediction higher:


```python
an_old_abalone = "15 1:1 2:0.655 3:0.53 4:0.175 5:1.2635 6:0.486 7:0.2635 8:0.415"
predict_and_plot(predictor, an_old_abalone)
```


    
![png](images/output_9_0.png)
    


## Delete Endpoint

Run the `delete_endpoint` to remove the hosted endpoint and avoid any charges from a stray instance being left on:


```python
predictor.delete_endpoint()
```
