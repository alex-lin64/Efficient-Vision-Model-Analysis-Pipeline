# Efficient Vision Model Analysis Pipeline (EVMAP)

This pipeline is composed of current SOTA object detection and activity recognition models.  This project is completed under the guidance of Boston Fusion Corp and aids in their completion of a proposal funded by Darpa's BAA [Ecole](https://www.darpa.mil/program/environment-driven-conceptual-learning).  The pipeline aims to coalesce the difficulty of analyzing different off-the-shelf models by providing a single framework to quickly switch models and datasets in order to get to the important task: gathering data.

Tool Stack:
- Triton Inference Server
  - NVidia support & documentation
  - Fast inference
  - Python API
- CVAT
  - Image and video annotations
  - Open source
  - Easy to set up 
- Fiftyone
  - Dataset visualization
  - Excellent integration with CVAT
  - Customizable to dataset needs


Below is a current system diagram of the baseline.  

![Baseline System Diagram](/readme_resources/baseline_sys_diagram.png)


Below you will find installation instructions and an overview of the typical workflow through the baseline pipeline.  A FAQ section at the bottom tracks common errors and bugs that may arise during the setup phase.


## Installation:

<details><summary> <b>Expand</b> </summary>

``` shell 

# clone the repository locally
git clone git@git.bostonfusion.com:bfc/operations/multi-domain-systems/arclight/baseline_object_detection.git

# go to the project folder
cd /baseline_object_detection

# install required packages in conda environment w/ python=3.8
pip install -r requirements.txt

# install ffmpeg (needed for fiftyone to work with videos), download from following link
https://ffmpeg.org/download.html

```

</details>

## Integrating Custom Models

> [!IMPORTANT]
> CUDA >= 11.7 is required to run the Triton Inference Server

Additional models added to the Triton Inference Server needs to be converted into the TensorRT engine format.  ONNX framework serves as an intermediary between pytorch and TensorRT enginer.  A workflow would be to export pytorch/tensorflow models in ONNX format, then export that through a docker container of the triton inference server as the TensorRT engine.

Example.

```bash
# 1. Export pytorch/tensorflow model as ONNX, usually a script in most model clones.  Please note that many conversion scripts to onnx require onnx-graphsurgeon, which is only compatible on linux.

# 2. ONNX -> TensorRT with trtexec and docker
docker run -it --rm --gpus=all nvcr.io/nvidia/tensorrt:22.06-py3

# 3. Copy onnx -> container: docker cp yolov7.onnx <container-id>:/workspace/

# 4. Export with FP16 precision, min batch 1, opt batch 8 and max batch 8
./tensorrt/bin/trtexec --onnx=yolov7.onnx --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --fp16 --workspace=4096 --saveEngine=yolov7-fp16-1x8x8.engine --timingCacheFile=timing.cache

# 5. Test engine
./tensorrt/bin/trtexec --loadEngine=yolov7-fp16-1x8x8.engine

# 6. Copy engine -> host: docker cp <container-id>:/workspace/yolov7-fp16-1x8x8.engine .

#7. The baseline gitlab contains the /triton_deploy directory prepopulated with the detectron2 and yolov7 tensorrt engines.  Adding new tensorrt engines must adhere to the repository structure

$ tree triton-deploy/
triton-deploy/
└── models
    └── yolov7
        ├── 1
        │   └── model.plan
        └── config.pbtxt
    └── your_model
        ├── 1
        │   └── model.plan
        └── config.pbtxt

# Create folder structure 
mkdir -p triton-deploy/models/your_model/1/  # 1 indicates version #
touch triton-deploy/models/your_model/config.pbtxt
# Place model
mv your_model.engine triton-deploy/models/your_model/1/model.plan

# steps 2-7 credit to yolov7 deploy readme
```

Once the model repository is extended with the new model, populate the config file.  The array of input/s and output/s must **EXACTLY** match the variable name and dims that the model was exported with.  These input and output names and dims can by uploading the model's onnx file to [Netron](https://netron.app/).  Additional configurations for tensorrt engines are documented [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).

``` shell
name: "your_model"
platform: "tensorrt_plan"
max_batch_size: 1
input [
  {
    name: "input_name"
    data_type: input_type
    dims: [input_dims]
  }
  ...
]
output [
  {
    name: "output_names"
    data_type: output_type
    dims: [output_dims]
  },
  ...
]
```

A client class must be implemented for the new model.  To simplify the manual work of connecting to the Triton Inference Server, base_triton_client.py is provided as a superclass to take care of the housekeeping involved with establishing server connection, inferencing, and exports.  The user is only responsible for implementing model-specific initialization values and two functions (listed below).  The new client should be organized into a new folder along with any supporting files.

```python

def _preprocess(self, input_image, inf_shape):
        """
        Preprocessing function, implemented in child class, performs necessary 
        modifications to input image for inference on the chosen model  
        
        :params:
            - input_image: the image to be processed
            - input_shape: [weight, height], image dimensions specified

        :returns:
            - the processed image ready for inference
        """
        # TODO: IMPLEMENT

    def _postprocess(self, results, input_image, inf_shape, tags):
        """
        Postprocessing function, implemented in child class, performs necessary 
        modifications to output data to get ready for visualization 

        NOTE: numeric outputs must be in float or float64 format

        :params:
            - results: raw inference data from the inference server 
            - input_image: original image used for inference
            - inf_shape: [inf_weight, inf_height], image dimensions specified for 
                inference
            - tags: list of tags to organize inference results in fiftyone
                
        :returns:
            - processed results in format
                {
                    "detections": [
                        {
                            "bbox": [<top-left-x>, <top-left-y>, <width>, <height>],
                            "label": obj_class
                            "confidence": 0.00-1.00
                        }
                    ],
                    "tags": [training/validation/testing]
                }
        """
        # TODO: IMPLEMENT
```

Once the client is implemented, front_end.ipynb step 4b should be modified to include the client in client_list as well as a new elif statement to initialize the new model client. 


> [!NOTE] 
> Examples can be seen within yolov7_triton_client.py or dectron2_triton_client.py within their respective client directories in client/.  There is also a template available to use in client/templates/ 





## Initialization: Triton Inference Server

``` shell
docker run --gpus all --rm --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 --mount type=bind,source="path/to/triton_deploy/models",destination=/models nvcr.io/nvidia/tritonserver:22.06-py3 tritonserver --model-repository=/models --strict-model-config=false --log-verbose 1

```

In the log you should see:

```
+------------+---------+--------+
| Model      | Version | Status |
+------------+---------+--------+
| yolov7     | 1       | READY  |
+------------+---------+--------+
| detectron2 | 1       | READY  |
+------------+---------+--------+
| ...        | 1       | READY  |
+------------+---------+--------+
```

The inference server will be reachable via port 8001.


## Initialization: CVAT

_optional (but not right now). CVAT is locally (until the ARCLIGHT server is up to host the application)

Follow the official CVAT [installation guide](https://opencv.github.io/cvat/docs/administration/basics/installation/).  

After installation, start CVAT with 

```shell

cd /path/to/cvat/

# to start
CVAT_VERSION=v2.3.0 docker compose up -d

# to stop (has to be within cvat directory)
docker compose down
```

CVAT can be accessed locally at localhost:8080


## Usage

> [!IMPORTANT]
> Datasets in Fiftyone can only consist of a single media type, either images, text, or video.  As such, when inferencing on a directory of videos or images, please ensure the directory contains only one type of media.


Refer to front_end.ipynb for more detailed instructions (step # refers to steps listed in front_end.ipynb)

1. (step 1) Import dependencies
2. (step 2) Start Triton Inference Server and CVAT (if working locally)
3. (step 3) Start Fiftyone instance locally
4. (step 3, optional) Upload pre-labeled data to Fiftyone
5. (step 4) Set client configs and select model to use
6. (step 4, optional) Select dataset to use
7. (step 4) Set inference configs and begin inference in desired mode
8. (Step 6) Run analysis metric (mAP, PR curves, f1-score) on the selected dataset before human validation
9. (step 5) Set classes, [label schema](https://docs.voxel51.com/user_guide/annotation.html), and configs for validation
10. (step 5) Run CVAT for human assisted validation (currently the metric used to select images with confidence < 60% for validation, changing bboxes, classes, adding attributes, etc.)
11. (step 5) Merge validated data back into original dataset in Fiftyone
12. (step 6) Run analysis metrics (mAP, PR curves, f1-score) on the selected dataset after human validation

  
If running everything locally, the tools used can be connected to at

* CVAT = localhost:8080
* Fiftyone = localhost:5151
* Triton = localhost:8001


> [!NOTE]
> As of now, images/videos used in the baseline are stored locally.  This means the links to the images are tied to each sample in Fiftyone are the location of the image on your local machine.  For consistency, use the data/ directory in baseline_system/ to store all iamges/videos required for use to avoid issues related to bad links.  **Ideally, all data should be stored in a dedicated directory on the ARCLIGHT server**.


## FAQ:

    Q: Why is requirements.txt so long?
    A: pip freeze was used, so it includes standard packages that are already installed in the base environment.  It should still work, but the requirements can be cleaned up by future editors.

    Q: Triton Inference Server starts, but the models are not ready
    A: This is usually a problem with the config files.  The input and output names and dims have to EXACTLY match the ones the model was serialized with.  Missing commas and closing brackets can also cause the error. 

    Q: Fiftyone stuck on error screen after deleting a dataset
    A: This is due to the way fiftyone handles deletes and exceptions.  Just restart fiftyone by rerunning the cell containing start_session().

    Q: Triton Inference Server displays 'defaulting to cpu' during inference
    A: The server is still using gpu, this can be verified through the Resource Monitor on windows or similar app on another OS.

    Q: Conda environments not allowing downloads through pip
    A: Run "python -m pip install --upgrade --force-reinstall pip" inside the conda env

    Q: Running validation on CVAT gives "bad gateway" error
    A: login to CVAT at localhost:8080 before running validation script, and make sure login info is correct

