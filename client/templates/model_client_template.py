import numpy as np
import sys
import cv2

from ..base_triton_client import Base_Inference_Client


class Model_Client_Template(Base_Inference_Client):
    """
    TODO: IMPLEMENT THE MODEL CLIENT FOR A NEW MODEL 
    """

    def __init__(
        self,
        url='localhost:8001',
        model_info=False,
        verbose=False,
        client_timeout=None,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None,
        width=640,
        height=640
    ) -> None:
        """
        TODO: SET THE DEFAULT VALUES FOR THE MODEL CLIENT PARAMS

        :params:
            - url: Inference server URL, default localhost:8001
            - model_info: Print model status, configuration and statistics
            - verbose: Enable verbose client output
            - client_timeout: Client timeout in seconds, default no timeout
            - ssl: Enable SSL encrypted channel to the server
            - root_certificates: File holding PEM-encoded root certificates, default none
            - private_key: File holding PEM-encoded private key, default is none
            - certificate_chain: File holding PEM-encoded certicate chain default is none
            - client_timeout: Client timeout in seconds, default no timeout
            - width: Inference model input width, default 640
            - height: Inference model input height, default 640
        """
        self.model = "MODEL_NAME"
        #  dict of names for the model input,  name:> type
        #   this should match the names of inputs and outputs of the config file 
        # TODO: POPULATE WITH MODEL INPUT_NAMES:INPUT_TYPE
        self.input_names = {}
        #  array of names for the model output
        #   this should match the names of inputs and outputs of the config file 
        # TODO: POPULATE WITH MODEL OUTPUT NAMES
        self.output_names = []
        super().__init__(
            input_names=self.input_names,
            output_names=self.output_names,
            model=self.model,
            url=url,
            model_info=model_info,
            verbose=verbose,
            client_timeout=client_timeout,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain,
            width=width,
            height=height
            )

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