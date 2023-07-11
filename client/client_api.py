#!/usr/bin/env python

import numpy as np
import sys
import cv2

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from .processing import preprocess, postprocess
from .render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from .labels import COCOLabels


class BaselineClient():

    def __init__(
        self,
        model='yolov7',
        url='localhost:8001',
        model_info=False,
        verbose=False,
        client_timeout=None,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None
    ) -> None:
        """
        ::params::
            - model -- Inference model name, default yolov7
            - url -- Inference server URL, default localhost:8001
            - model_info -- Print model status, configuration and statistics
            - verbose -- Enable verbose client output
            - client_timeout -- Client timeout in seconds, default no timeout
            - ssl -- Enable SSL encrypted channel to the server
            - root_certificates -- File holding PEM-encoded root certificates, default none
            - private_key -- File holding PEM-encoded private key, default is none
            - certificate_chain -- File holding PEM-encoded certicate chain default is none
        """
        self.model = model
        self.__url = url
        self.__model_info = model_info
        self.__verbose = verbose
        self.__client_timeout = client_timeout
        self.__ssl = ssl
        self.__root_certificates = root_certificates
        self.__private_key = private_key
        self.__certificate_chain = certificate_chain
        self.__INPUT_NAMES = ["images"]
        self.__OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]

        # create server context
        try:
            self.__triton_client = grpcclient.InferenceServerClient(
                url=self.__url,
                verbose=self.__verbose,
                ssl=self.__ssl,
                root_certificates=self.__root_certificates,
                private_key=self.__private_key,
                certificate_chain=self.__certificate_chain
            )
        except Exception as e:
            print("Context creation failed: " + str(e))
            sys.exit()
        
        # server health check
        if not self.__triton_client.is_server_live():
            print("FAILED: is_server_live")
            sys.exit(1)

        if not self.__triton_client.is_server_ready():
            print("FAILED: is_server_ready")
            sys.exit(1)

        if not self.__triton_client.is_model_ready(self.model):
            print("FAILED: is_model_ready")
            sys.exit(1)
        
        if self.__model_info:
            try:
                metadata = self.__triton_client.get_model_metadata(self.model)
                print(metadata)
            except InferenceServerException as ex:
                if "Request for unknown model" not in ex.message():
                    print("FAILED : get_model_metadata")
                    print("Got: {}".format(ex.message()))
                    sys.exit(1)
                else:
                    print("FAILED : get_model_metadata")
                    sys.exit(1)

            # Model configuration
            try:
                config = self.__triton_client.get_model_config(self.model)
                if not (config.config.name == self.model):
                    print("FAILED: get_model_config")
                    sys.exit(1)
                print(config)
            except InferenceServerException as ex:
                print("FAILED : get_model_config")
                print("Got: {}".format(ex.message()))
                sys.exit(1)

    def inference(
            self,
            input_,
            output_ = '',
            mode='dummy',
            width=640,
            height=640,
            fps=24.0
    ):
        """
        Runs inference on the triton inference server with the model specified 
        for this baseline client

        ::param::
            - input_ -- Input file to load from in image or video mode
            - output -- Write output into file instead of displaying it
            - mode -- Run mode. \'dummy\' will send an emtpy buffer to the server to test if 
              inference works. \'image\' will process an image. \'video\' will process a video.'
            - width -- Inference model input width, default 640
            - height -- Inference model input height, default 640
            - fps -- Video output fps, default 24.0 FPS
        """
        inputs = []
        outputs = []

        inputs.append(grpcclient.InferInput(self.__INPUT_NAMES[0], [1, 3, width, height], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput(self.__OUTPUT_NAMES[0]))
        outputs.append(grpcclient.InferRequestedOutput(self.__OUTPUT_NAMES[1]))
        outputs.append(grpcclient.InferRequestedOutput(self.__OUTPUT_NAMES[2]))
        outputs.append(grpcclient.InferRequestedOutput(self.__OUTPUT_NAMES[3]))

        if mode == 'dummy':
            print("Running in 'dummy' mode")
            print("Creating emtpy buffer filled with ones...")
            self.__infer_dummy(inputs, outputs, width, height)
        elif mode == 'image':
            print("Running in 'image' mode")
            if not input_:
                print("FAILED: no input image")
                sys.exit(1)
            self.__infer_image(input_, output_, width, height, inputs, outputs)
        elif mode == 'video':
            print("Running in 'video' mode")
            if not input_:
                print("FAILED: no input video")
                sys.exit(1)
            self.__infer_video(input_, output_, fps, width, height, inputs, outputs)
        else:
            print(f"mode error, refer to instructions: \'dummy\' will send an \
                  emtpy buffer to the server to test if inference works. \'image\' \
                  will process an image. \'video\' will process a video")
            sys.exit(1)

    
    def __infer_image(self, input_, output_, width, height, inputs, outputs):
        """
        Processes an image through the inference server, performs inference on it
        and displays (or saves) the processed results

        ::param::
            - input_ -- Input file to load from in image or video mode
            - output -- Write output into file instead of displaying it
            - width -- Inference model input width, default 640
            - height -- Inference model input height, default 640
            - inputs -- array of Triton Input objects to store the input for inference
            - outputs -- array of Triton Output objects to store the output of inference

        ** Note: detecting using coco labels
        """
        print("Creating buffer from image file...")
        input_image = cv2.imread(str(input_))
        if input_image is None:
            print(f"FAILED: could not load input image {str(input_)}")
            sys.exit(1)
        input_image_buffer = preprocess(input_image, [width, height])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)

        inputs[0].set_data_from_numpy(input_image_buffer)

        print("Invoking inference...")
        results = self.__triton_client.infer(
            model_name=self.model,
            inputs=inputs,
            outputs=outputs,
            client_timeout=self.__client_timeout
        )

        if self.__model_info:
            self.__get_model_stats()
        print("Done")

        for output in self.__OUTPUT_NAMES:
            res = results.as_numpy(output)
            print(f"Received result buffer \"{output}\" of size {res.shape}")
            print(f"Naive buffer sum: {np.sum(res)}")
        
        num_dets = results.as_numpy(self.__OUTPUT_NAMES[0])
        det_boxes = results.as_numpy(self.__OUTPUT_NAMES[1])
        det_scores = results.as_numpy(self.__OUTPUT_NAMES[2])
        det_classes = results.as_numpy(self.__OUTPUT_NAMES[3])
        detected_objects = postprocess(
            num_dets, 
            det_boxes, 
            det_scores, 
            det_classes, 
            input_image.shape[1], 
            input_image.shape[0], 
            [width, height])
        print(f"Detected objects: {len(detected_objects)}")

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

        if output_:
            cv2.imwrite(output_, input_image)
            print(f"Saved result to {output}")
        else:
            cv2.imshow('image', input_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def __infer_video(self, input_, output_, fps, width, height, inputs, outputs):
        """
        Processes a video through the inference server, performs inference on it
        and displays (or saves) the processed results

        ::param::
            - input_ -- Input file to load from in image or video mode
            - output -- Write output into file instead of displaying it
            - fps -- Video output fps, default 24.0 FPS
            - width -- Inference model input width, default 640
            - height -- Inference model input height, default 640
            - inputs -- array of Triton Input objects to store the input for inference
            - outputs -- array of Triton Output objects to store the output of inference

        ** Note: detecting using coco labels
        """
        print("Opening input video stream...")
        cap = cv2.VideoCapture(input_)
        if not cap.isOpened():
            print(f"FAILED: cannot open video {input_}")
            sys.exit(1)
        
        counter = 0
        out = None
        print("Invoking inference...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("failed to fetch next frame")
                break
            
            if counter == 0 and output_:
                print("Opening output video stream...")
                fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
                out = cv2.VideoWriter(output_, fourcc, fps, (frame.shape[1], frame.shape[0]))

            input_image_buf = preprocess(frame, [width, height])
            input_image_buf = np.expand_dims(input_image_buf, axis=0)
            inputs[0].set_data_from_numpy(input_image_buf)

            results = self.__triton_client.infer(
                model_name=self.model,
                inputs=inputs,
                outputs=outputs,
                client_timeout=self.__client_timeout
            )

            num_dets = results.as_numpy("num_dets")
            det_boxes = results.as_numpy("det_boxes")
            det_scores = results.as_numpy("det_scores")
            det_classes = results.as_numpy("det_classes")
            detected_objs = postprocess(
                num_dets,
                det_boxes,
                det_scores,
                det_classes,
                frame.shape[1],
                frame.shape[0],
                [width, height]
            )

            print(f"Frame {counter}: {len(detected_objs)} objects")
            counter += 1

            for box in detected_objs:
                # CURRENTLY USES COCO LABELS
                print(f"{COCOLabels(box.classID).name}: {box.confidence}")
                frame = render_box(
                    frame, box.box(), 
                    color=tuple(RAND_COLORS[box.classID % 64].tolist())
                    )
                size = get_text_size(
                    frame, 
                    f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", 
                    normalised_scaling=0.6
                    )
                frame = render_filled_box(
                    frame, 
                    (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]), 
                    color=(220, 220, 220)
                    )
                frame = render_text(
                    frame, 
                    f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", 
                    (box.x1, box.y1), 
                    color=(30, 30, 30), 
                    normalised_scaling=0.5
                    )

            if output_:
                out.write(frame)
            else:
                cv2.imshow('image', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        
        if self.__model_info:
            self.__get_model_stats()
        print("Done.")

        cap.release()
        if output_:
            out.release()
        else:
            cv2.destroyAllWindows()

    def __infer_dummy(self, inputs, outputs, width, height):
        """
        Sends an empty buffer to triton inference server as test of functionality

        ::param::
            - inputs -- array of Triton Input objects to store the input for inference
            - outputs -- array of Triton Output objects to store the output of inference
            - width -- Inference model input width, default 640
            - height -- Inference model input height, default 640
        """
        inputs[0].set_data_from_numpy(np.ones(shape=(1,3,width,height), dtype=np.float32))
        
        print("Invoking inference...")
        results = self.__triton_client.infer(
                model_name=self.model,
                inputs=inputs,
                outputs=outputs,
                client_timeout=self.__client_timeout
            )

        if self.__model_info:
            self.__get_model_stats()
        print("Done.")

        for output in self.__OUTPUT_NAMES:
            res = results.as_numpy(output)
            print(f"Recieved result buffer \"{output}\" of size {res.shape}")
            print(f"Naive buffer sum: {np.sum(res)}")

    def __get_model_stats(self):
        """
        Gets the model stats from the inference client
        """
        stats = self.__triton_client.get_inference_statistics(model_name=self.model)
        if len(stats.model_stats) != 1:
            print("FAILED: get_inference_statistics")
            sys.exit(1)
        print(stats)
