import numpy as np
import sys
import cv2

from ..base_triton_client import Base_Inference_Client
from .processing import preprocess, postprocess
from .render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from .labels import COCOLabels


class YoloV7_Triton_Inference_Client(Base_Inference_Client):
    """
    Implements inference for yolov7 tensorrt engine 
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
        self.model = "yolov7"
        #  dict of names for the model input,  name:> type
        #   this should match the names of inputs and outputs of the config file 
        self.input_names = {"images":"FP32"}
        #  array of names for the model output
        #   this should match the names of inputs and outputs of the config file 
        self.output_names = ["num_dets", "det_boxes", "det_scores", "det_classes"]
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
        return preprocess(input_image, inf_shape)

    def _postprocess(self, results, input_image, inf_shape, tags):
        """
        Postprocessing function, implemented in child class, performs necessary 
        modifications to output data to get ready for visualization 

        NOTE: numeric outputs must be in float or float64 format

        NOTE: yolov7 is trained on COCOClasses

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
        num_dets = results.as_numpy(self.output_names[0])
        det_boxes = results.as_numpy(self.output_names[1])
        det_scores = results.as_numpy(self.output_names[2])
        det_classes = results.as_numpy(self.output_names[3])
        img_h, img_w = input_image.shape[0], input_image.shape[1]

        # collect raw inference data from output tensors
        boxes = det_boxes[0, :num_dets[0][0]] / np.array([inf_shape[0], inf_shape[1], inf_shape[0], inf_shape[1]], dtype=np.float32)
        scores = det_scores[0, :num_dets[0][0]]
        classes = det_classes[0, :num_dets[0][0]].astype(np.int32)

        old_h, old_w = img_h, img_w
        offset_h, offset_w = 0, 0

        if (img_w / inf_shape[1]) >= (img_h / inf_shape[0]):
            old_h = int(inf_shape[0] * img_w / inf_shape[1])
            offset_h = (old_h - img_h) // 2
        else:
            old_w = int(inf_shape[1] * img_h / inf_shape[0])
            offset_w = (old_w - img_w) // 2

        boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
        boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
        boxes = boxes.astype(np.int32)

        annotation = {}
        detections = []
        for i in range(num_dets[0][0]):
            det = {}
            det["bbox"] = [
                boxes[i][0] / img_w, 
                boxes[i][1] / img_h, 
                (boxes[i][2] - boxes[i][0]) / img_w, 
                (boxes[i][3] - boxes[i][1]) / img_h
                ]
            det["label"] = COCOLabels(classes[i]).name
            det["confidence"] = float(scores[i]) 
            detections.append(det)

        annotation["detections"] = detections
        annotation["tags"] = tags

        return annotation

    def _visualize(self, results, input_image, inf_shape):
        """
        Composes the processed results onto the input image to create visual 
        representation of the detections

        :params:
            - results: raw inference data from the inference server 
            - input_image: original image input 
            - inf_shape: [inf_weight, inf_height], image dimensions specified for 
                inference

        :returns:
            - the input image with the detecitons visualized 

        ** NOTE: this yolov7 is trained on COCO labels
        """
        num_dets = results.as_numpy(self.output_names[0])
        det_boxes = results.as_numpy(self.output_names[1])
        det_scores = results.as_numpy(self.output_names[2])
        det_classes = results.as_numpy(self.output_names[3])
        detected_objects = postprocess(
            num_dets, 
            det_boxes, 
            det_scores, 
            det_classes, 
            input_image.shape[1], 
            input_image.shape[0], 
            inf_shape)
        
        for box in detected_objects:
            print(f"{COCOLabels(box.classID).name}: {box.confidence}")
            input_image = render_box(
                input_image, 
                box.box(), 
                color=tuple(RAND_COLORS[box.classID % 64].tolist())
                )
            size = get_text_size(
                input_image, 
                f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", 
                normalised_scaling=0.6
                )
            input_image = render_filled_box(
                input_image, 
                (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]), 
                color=(220, 220, 220)
                )
            input_image = render_text(
                input_image, 
                f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", 
                (box.x1, box.y1), 
                color=(30, 30, 30), 
                normalised_scaling=0.5
                )
        
        return input_image
