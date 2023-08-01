#!/usr/bin/env python

from abc import abstractmethod
from os.path import isfile, isdir
import glob
import numpy as np
import sys
import cv2
import os

import fiftyone as fo
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class Base_Inference_Client():
    """
    A base class implementing the triton grpc client for the Triton Inference 
    Server for custom model image/video/dummy inference from the Triton Inference 
    Server and for models in format of TensorRT engines

    Contains three functions to be implemented

        preprocess() \
        postprocess() \
        visualize() \
    """

    def __init__(
        self,
        input_names,
        output_names,
        model,
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
            - input_names: list of input names, must exactly match the name and
                number of inputs specificed in model's config
            - output_names: lists of output names, must exactly match the name and
                number of inputs specificed in model's config
            - model: name of the model as specified in the config
            - url: Inference server URL, default localhost:8001
            - model_info: Print model status, configuration and statistics
            - verbose: Enable verbose client output
            - client_timeout: Client timeout in seconds, default no timeout
            - ssl: Enable SSL encrypted channel to the server
            - root_certificates: File holding PEM-encoded root certificates, default none
            - private_key: File holding PEM-encoded private key, default is none
            - certificate_chain: File holding PEM-encoded certicate chain default is none
            - client_timeout: Client timeout in seconds, default no timeout
            - width: Inference model input width
            - height: Inference model input height
        """
        # grpcclient infererence object for input to inference server
        self.server_inputs = []
        # grpcclient infererence object for ouputs to inference server
        self.server_outputs = []
        self.input_names = input_names
        self.output_names = output_names
        self.model = model
        self.__url = url
        self.__model_info = model_info
        self.__verbose = verbose
        self.__ssl = ssl
        self.__root_certificates = root_certificates
        self.__private_key = private_key
        self.__certificate_chain = certificate_chain
        self.__client_timeout = client_timeout
        self.inf_width = width
        self.inf_height = height

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

        # init the input/output object arrays
        self.__init_input_output(input_names, output_names)

    def __init_input_output(self, input_names, output_names):
        """
        Initializes the input and output objects needed to inference from Triton
        Inference Server

        :params:
            - input_names: dict of input names to input type, must exactly match the name and
                number of inputs specificed in model's config
            - output_names: lists of output names, must exactly match the name and
                number of inputs specificed in model's config
        """
        # init input objects
        for input_name in input_names:
            self.server_inputs.append(grpcclient.InferInput(
                input_name, 
                [1, 3, self.inf_width, self.inf_height], 
                input_names[input_name]
                ))
            
        # init output objects
        for output_name in output_names:
            self.server_outputs.append(grpcclient.InferRequestedOutput(output_name))

    def infer_image(self, input_, vis=False, output_='', fo_dataset='', tags=[]):
        """
        Processes an image through the inference server, performs inference on it
        and displays (or saves) the processed results

        :param:
            - input_: Input directory to load from in image
                NOTE: directory must only contain image files
            - fo_dataset: Dataset name to export predictions to fiftyone, 
                default '', no export
            - vis: Show visualization of prediction on computer, default false
            - output_: Output directory, default no output saved
            - tags: list of tags to organize inference results in fiftyone
        """
        # input check
        print("Running in 'image' mode")
        if not input_:
            print("FAILED: no input image/s")
            sys.exit(1)
        
        # read in images files
        filenames = self.__read_input(input_)

        # print model stats 
        if self.__model_info:
            self.__get_model_stats()
        print("Done")

        annotations = {}
        for i, filename in enumerate(filenames):
            # existence check
            print("Creating buffer from image file...")
            input_image = cv2.imread(str(filename.path))
            if input_image is None:
                print(f"FAILED: could not load input image {str(filename.path)}")
                continue

            # preprocess input image
            input_image_buffer = self._preprocess(input_image, [self.inf_width, self.inf_height])

            # set server input to the preprocessed input image
            self.server_inputs[0].set_data_from_numpy(input_image_buffer)

            print("Invoking inference...")
            results = self.__triton_client.infer(
                model_name=self.model,
                inputs=self.server_inputs,
                outputs=self.server_outputs,
                client_timeout=self.__client_timeout
            )

            # print data recieved from the server
            for output in self.output_names:
                res = results.as_numpy(output)
                print(f"Received result buffer \"{output}\" of size {res.shape}")
                print(f"Naive buffer sum: {np.sum(res)}")
            
            # post process raw data from server
            annotation = self._postprocess(results, input_image, [self.inf_width, self.inf_height], tags)
            print(f"Detected objects: {len(annotation)}")

            # saves detections in annotations format described in method description
            #   for use in export_to_fo
            annotations[filename.path] = annotation

            # writes the image out/ displays it
            if output_:
                # visualize the post processed data
                final_image = self._visualize(results, input_image, [self.inf_width, self.inf_height])
                cv2.imwrite(os.path.join(output_, f"res_img_{i}.jpg"), final_image)
                print(f"Saved result to {output_}")
            if vis:
                # visualize the post processed data
                final_image = self._visualize(results, input_image, [self.inf_width, self.inf_height])
                cv2.imshow('image', final_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # upload data to fo
        if fo_dataset:
            self.__export_image_to_fo(annotations, dataset=fo_dataset)

    def infer_video(self, input_, vis=False, fo_dataset='', output_='', fps=24.0, tags=[]):
        """
        Processes a video through the inference server, performs inference on it
        and displays (or saves) the processed results

        :param:
            - input_: Input directory to load from in video
                NOTE: directory must only contain video files
            - vis: Show visualization of prediction on computer, default false
            - fo_dataset: Dataset name to export predictions to fiftyone, 
                default '', no export
            - output_: Output directory, default no output saved
            - fps: Video output fps, default 24.0 FPS
            - tags: list of tags to organize inference results in fiftyone
        """
        # input check
        print("Running in 'video' mode")
        if not input_:
            print("FAILED: no input video")
            sys.exit(1)
        
        # read in images files
        filenames = self.__read_input(input_)

        # gets the model statistics from server
        if self.__model_info:
            self.__get_model_stats()
        print("Done.")

        annotations = {}
        for i, filename in enumerate(filenames):
            # existence check
            print("Opening input video stream...")
            cap = cv2.VideoCapture(filename.path)
            if not cap.isOpened():
                print(f"FAILED: cannot open video {filename.path}")
                continue

            annotations[filename.path] = {}

            n_frame = 1
            out = None   # file location to write output to
            print("Invoking inference...")
            # start loop to inference on video
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("failed to fetch next frame")
                    break
                
                # open video stream
                if not n_frame and output_:
                    print("Opening output video stream...")
                    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
                    out = cv2.VideoWriter(
                        os.path.join(output_, f"res_vid_{i}.mp4"), 
                        fourcc, 
                        fps, 
                        (frame.shape[1], frame.shape[0])
                        )

                # preprocess
                input_image_buf = self._preprocess(frame, [self.inf_width, self.inf_height])
                self.server_inputs[0].set_data_from_numpy(input_image_buf)

                # run inference on the current frame
                results = self.__triton_client.infer(
                    model_name=self.model,
                    inputs=self.server_inputs,
                    outputs=self.server_outputs,
                    client_timeout=self.__client_timeout
                    )

                # run postprocessing
                annotation = self._postprocess(results, frame, [self.inf_width, self.inf_height], tags)
                print(f"Frame {n_frame}: {len(annotation)} objects")
                # add annotation of frame under the video 
                annotations[filename.path][n_frame] = annotation 
                # increment frame
                n_frame += 1
                
                # writes the image out/ displays it
                if output_:
                    # visualize the post processed data
                    final_image = self._visualize(results, frame, [self.inf_width, self.inf_height])
                    out.write(final_image)
                if vis:
                    # visualize the post processed data
                    final_image = self._visualize(results, frame, [self.inf_width, self.inf_height])
                    cv2.imshow('video', final_image)
                    if cv2.waitKey(1) == ord('q'):
                        break

            # clean up video stream
            cap.release()
            if output_:
                out.release()
            else:
                cv2.destroyAllWindows()
        
        if fo_dataset:
            self.__export_video_to_fo(annotations, fo_dataset)

    def infer_dummy(self):
        """       
        Sends an empty buffer to triton inference server as test of functionality
        """
        print("Running in 'dummy' mode")
        print("Creating emtpy buffer filled with ones...")

        # creates buffer image
        self.server_inputs[0].set_data_from_numpy(
            np.ones(shape=(1,3,self.inf_width,self.inf_height), dtype=np.float32)
            )
        
        # runs buffer inference
        print("Invoking inference...")
        results = self.__triton_client.infer(
                model_name=self.model,
                inputs=self.server_inputs,
                outputs=self.server_outputs,
                client_timeout=self.__client_timeout
                )

        # get model stats from server
        if self.__model_info:
            self.__get_model_stats()
        print("Done.")

        print(self.server_inputs)

        # print out buffer inference results
        for output in self.output_names:
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

    def __read_input(self, input_):
        """
        Reads the input files names to list, skipping directories

        :params:
            - input_: directory to load from in image or video mode

        :returns:
            - array of scandir objects containing attributes for filename and path
        """
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".mov", ".avi"]
        
        # NOTE currently, doesn't distinguish between video and images in folder, 
        #   up to user to prepare proper inference directory
        # loop through directory and append every image
        filenames = []
        for filename in os.scandir(str(input_)):
            # skip directories and symlinks, only add images
            if (filename.is_file(follow_symlinks=False) and \
                os.path.splitext(filename.path)[1].lower() in extensions): 
                filenames.append(filename)

        return filenames

    def __export_image_to_fo(self, annotations, dataset):
        """
        Transforms image detection data into format for fiftyone as uploads to 
        existing dataset or as a new dataset

        Annotations format is 
        {
            "path/to/image": {
                "detections": [
                    {
                        "bbox": [<top-left-x>, <top-left-y>, <width>, <height>],
                        "label": obj_class
                        "confidence": 0.00-1.00
                    }
                ],
                "tags": [training/validation/testing]
            }
        }

        :params:
            - annotations: dict of images and their filepaths to model predictions
            - dataset: name of dataset to export annotations to in fiftyone, if 
                dataset name already exists in fiftyone, will append to dataset
        """
        samples = []

        for filepath in annotations:
            sample = fo.Sample(filepath=filepath)

            detections = []
            # convert all detections to fo format
            for det in annotations[filepath]['detections']:
                label = det['label']
                # bbox coordingates in format [top-left-x, top-left-y, width, height]
                bbox = det['bbox']
                confidence = det['confidence']

                detections.append(
                    fo.Detection(label=label, bounding_box=bbox, confidence=confidence)
                )

            # set sample detections to converted samples
            sample['model_detections'] = fo.Detections(detections=detections)
            sample.tags = annotations[filepath]["tags"]
            samples.append(sample)
                           
        # check if dataset is in list
        if dataset in fo.list_datasets():
            dataset = fo.load_dataset(dataset)
        else:
            print("Dataset does not exist in fiftyone, creating new \
                  dataset by default.  Make sure the dataset set to persistent \
                  if using existing fiftyone datset")
            dataset = fo.Dataset(dataset)
        
        # set dataset to persistent, stay in fo
        dataset.persistent = True
        # populate dataset with the processed samples
        dataset.add_samples(samples)

    def __export_video_to_fo(self, annotations, dataset):
        """
        Transforms video detection data into format for fiftyone as uploads to 
        existing dataset or as a new dataset

        Annotations format is 
        {
            "path/to/video": {
                1: {
                    "detections": [
                        {
                            "bbox": [<top-left-x>, <top-left-y>, <width>, <height>],
                            "label": obj_class
                            "confidence": 0.00-1.00
                        }
                    ],
                    tags": [training/validation/testing]
                }
                2: { ... }
            }
        }

        :params:
            - annotations: dict of videos and their filepaths to model predictions 
                ordered by frame
            - dataset: name of dataset to export annotations to in fiftyone, if 
                dataset name already exists in fiftyone, will append to dataset
        """
        samples = []

        for filepath in annotations:
            sample = fo.Sample(filepath=filepath)

            # create video sample with frame labels
            for n_frame, annotation in annotations[filepath].items():
                detections = []
                frame = fo.Frame()

                for det in annotation["detections"]:
                    label = det["label"]
                    # bbox coordingates in format [top-left-x, top-left-y, width, height]
                    bbox = det['bbox']
                    confidence = det['confidence']

                    detections.append(
                        fo.Detection(label=label, bounding_box=bbox, confidence=confidence)
                    )
                # append detections in frame to the frame
                frame["objects"] = fo.Detections(detections=detections)
                # add frame to sample
                sample.frames[n_frame] = frame
            # add tag to the video
            sample.tags = annotation["tags"]
            samples.append(sample)
        
        # check if dataset is in list
        if dataset in fo.list_datasets():
            dataset = fo.load_dataset(dataset)
        else:
            print("Dataset does not exist in fiftyone, creating new \
                  dataset by default.  Make sure the dataset set to persistent \
                  if using existing fiftyone datset")
            dataset = fo.Dataset(dataset)
        
        # set dataset to persistent, stay in fo
        dataset.persistent = True
        # populate dataset with the processed samples
        dataset.add_samples(samples)


    @abstractmethod
    def _preprocess(self, input_image, inf_shape):
        """
        Preprocessing function, implemented in child class, performs necessary 
        modifications to input image for inference on the chosen model  
        
        :params:
            - input_image: the image to be processed
            - inf_shape: [inf_weight, inf_height], image dimensions specified for
                inference

        :returns:
            - the processed image ready for inference
        """

    @abstractmethod
    def _postprocess(self, results, input_image, inf_shape, tags, scale=None):
        """
        Postprocessing function, implemented in child class, performs necessary 
        modifications to output data to be uploaded to fiftyone

        :params:
            - results: raw inference data from the inference server 
            - input_image: original image used for inference
            - inf_shape: [inf_weight, inf_height], image dimensions specified for 
                inference
            - tags: list of tags to organize inference results in fiftyone
            - scales: image resize scale, default: no scale postprocessing applied

        :returns:
            - processed results ready to be uploaded to fiftyone
        """
    
    @abstractmethod
    def _visualize(self, results, input_image, inf_shape):
        """
        Processes results and composes the processed results onto the input image to create visual 
        representation of the detections

        :params:
            - results: raw inference data from the inference server 
            - input_image: original image input 
            - inf_shape: [inf_weight, inf_height], image dimensions specified for 
                inference

        :returns:
            - the input image with the detecitons visualized 
        """
    