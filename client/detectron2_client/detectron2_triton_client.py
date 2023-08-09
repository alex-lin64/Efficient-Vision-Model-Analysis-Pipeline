import os
import sys
import cv2
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from ..base_triton_client import Base_Inference_Client

# colors for visualization
COLORS = ['GoldenRod', 'MediumTurquoise', 'GreenYellow', 'SteelBlue', 'DarkSeaGreen', 'SeaShell', 'LightGrey',
          'IndianRed', 'DarkKhaki', 'LawnGreen', 'WhiteSmoke', 'Peru', 'LightCoral', 'FireBrick', 'OldLace',
          'LightBlue', 'SlateGray', 'OliveDrab', 'NavajoWhite', 'PaleVioletRed', 'SpringGreen', 'AliceBlue', 'Violet',
          'DeepSkyBlue', 'Red', 'MediumVioletRed', 'PaleTurquoise', 'Tomato', 'Azure', 'Yellow', 'Cornsilk',
          'Aquamarine', 'CadetBlue', 'CornflowerBlue', 'DodgerBlue', 'Olive', 'Orchid', 'LemonChiffon', 'Sienna',
          'OrangeRed', 'Orange', 'DarkSalmon', 'Magenta', 'Wheat', 'Lime', 'GhostWhite', 'SlateBlue', 'Aqua',
          'MediumAquaMarine', 'LightSlateGrey', 'MediumSeaGreen', 'SandyBrown', 'YellowGreen', 'Plum', 'FloralWhite',
          'LightPink', 'Thistle', 'DarkViolet', 'Pink', 'Crimson', 'Chocolate', 'DarkGrey', 'Ivory', 'PaleGreen',
          'DarkGoldenRod', 'LavenderBlush', 'SlateGrey', 'DeepPink', 'Gold', 'Cyan', 'LightSteelBlue', 'MediumPurple',
          'ForestGreen', 'DarkOrange', 'Tan', 'Salmon', 'PaleGoldenRod', 'LightGreen', 'LightSlateGray', 'HoneyDew',
          'Fuchsia', 'LightSeaGreen', 'DarkOrchid', 'Green', 'Chartreuse', 'LimeGreen', 'AntiqueWhite', 'Beige',
          'Gainsboro', 'Bisque', 'SaddleBrown', 'Silver', 'Lavender', 'Teal', 'LightCyan', 'PapayaWhip', 'Purple',
          'Coral', 'BurlyWood', 'LightGray', 'Snow', 'MistyRose', 'PowderBlue', 'DarkCyan', 'White', 'Turquoise',
          'MediumSlateBlue', 'PeachPuff', 'Moccasin', 'LightSalmon', 'SkyBlue', 'Khaki', 'MediumSpringGreen',
          'BlueViolet', 'MintCream', 'Linen', 'SeaGreen', 'HotPink', 'LightYellow', 'BlanchedAlmond', 'RoyalBlue',
          'RosyBrown', 'MediumOrchid', 'DarkTurquoise', 'LightGoldenRodYellow', 'LightSkyBlue']


# labels from COCO dataset
LABELS = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",
          "stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
          "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
          "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
          "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
          "cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
          "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
          "hair drier", "toothbrush"]



class Detectron2_Triton_Client(Base_Inference_Client):
    """
    Implements inference for detectron2 tensorrt engine.
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
        width=1344,
        height=1344,
        nms_threshold=None,
        iou_threshold=0.5
    ) -> None:
        """
        :params:
            - input_names: list of input names, must exactly match the name and
                number of inputs specificed in model's config
            - output_names: lists of output names, must exactly match the name and
                number of inputs specificed in model's config
            - url: Inference server URL, default localhost:8001
            - model_info: Print model status, configuration and statistics
            - verbose: Enable verbose client output
            - client_timeout: Client timeout in seconds, default no timeout
            - ssl: Enable SSL encrypted channel to the server
            - root_certificates: File holding PEM-encoded root certificates, default none
            - private_key: File holding PEM-encoded private key, default is none
            - certificate_chain: File holding PEM-encoded certicate chain default is none
            - client_timeout: Client timeout in seconds, default no timeout
            - width: Inference model input width, 1344 is required for Detectron2
            - height: Inference model input height, 1344 is required for Detectron2
            - nms_threshold: Threshold for non-max suppression, default None
            - iou_threshold: Threshold for intersection over union, default 0.5
        """
        self.model = "detectron2"
        #  dict of names for the model input,  name:> type
        #   this should match the names of inputs and outputs of the config file 
        self.input_names = {"input_tensor":"FP32"}
        self.nms_threshold = nms_threshold
        self.iou_threshold = iou_threshold
        # from the mask_rcnn_R_50_FPN_3x config file
        self.min_size_test = 800
        self.max_size_test = 1333
        # scaling for inference images
        self.scale = None
        #  array of names for the model output
        #   this should match the names of inputs and outputs of the config file 
        self.output_names = [
            "detection_boxes_box_outputs", 
            "detection_classes_box_outputs", 
            "detection_scores_box_outputs", 
            "num_detections_box_outputs", 
            "detection_masks"
            ]
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
        Preprocessing function, prepares it as needed for batching such as padding,
        resizing, normalization, data type casting, and transposing.
        
        :params:
            - input_image: the image to be processed
            - inf_shape: [inf_weight, inf_height], image dimensions specified for
                inference

        :returns:
            - the processed image ready for inference
        """

        def resize_pad(image, pad_color=(0, 0, 0)):
            """
            Resizes the image to fit fully within the input size, and pads the 
            remaining bottom-right portions with the value provided.

            :params:
                - param image: The PIL image object
                - pad_color: The RGB values to use for the padded area. Default: Black/Zeros.

            :returns:
                The image already padded and cropped, and the resize scale used.
            """

            # Get characteristics.
            width, height = image.size

            # Replicates behavior of ResizeShortestEdge augmentation.
            size = self.min_size_test * 1.0
            pre_scale = size / min(height, width)
            if height < width:
                newh, neww = size, pre_scale * width
            else:
                newh, neww = pre_scale * height, size

            # If delta between min and max dimensions is so that max sized dimension reaches self.max_size_test
            # before min dimension reaches self.min_size_test, keeping the same aspect ratio. We still need to
            # maintain the same aspect ratio and keep max dimension at self.max_size_test.
            if max(newh, neww) > self.max_size_test:
                pre_scale = self.max_size_test * 1.0 / max(newh, neww)
                newh = newh * pre_scale
                neww = neww * pre_scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)

            # Scaling factor for normalized box coordinates scaling in post-processing.
            scaling = max(newh/height, neww/width)

            # Padding.
            image = image.resize((neww, newh), resample=Image.BILINEAR)
            pad = Image.new("RGB", (inf_shape[0], inf_shape[1]))
            pad.paste(pad_color, [0, 0, inf_shape[0], inf_shape[1]])
            pad.paste(image)
            return pad, scaling

        # convert cv2 image format to Pillow format
        input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(input_image_rgb)
        # Pad with mean values of COCO dataset, since padding is applied before actual model's
        # preprocessor steps (Sub, Div ops), we need to pad with mean values in order to reverse
        # the effects of Sub and Div, so that padding after model's preprocessor will be with actual 0s.
        image, self.scale = resize_pad(image, (124, 116, 104))
        image = np.asarray(image, dtype=np.float32)
        # Change HWC -> CHW.
        image = np.transpose(image, (2, 0, 1))
        # Change RGB -> BGR.
        return np.expand_dims(image[[2,1,0]], axis=0)

    def _postprocess(self, results, input_image, inf_shape):
        """
        Postprocessing function, implemented in child class, performs necessary 
        modifications to output data to get ready for visualization 

        :params:
            - results: raw inference data from the inference server 
            - input_image: the image to be processed
            - inf_shape: [inf_weight, inf_height], image dimensions specified for
                inference
        :returns:
            - processed results ready for visualization
        """
        boxes = results.as_numpy(self.output_names[0])
        pred_classes = results.as_numpy(self.output_names[1])
        scores = results.as_numpy(self.output_names[2])
        nums = results.as_numpy(self.output_names[3])
        masks = results.as_numpy(self.output_names[4])

        print("boxes: ", boxes.shape)
        print("pred_classes: ", pred_classes.shape)
        print("masks: ", masks.shape)
        print("scores: ", scores.shape)
        print("nums: ", nums)

        print(scores)

        detected_objects = []
        for n in range(int(nums)):
            # Select a mask.
            mask = masks[0][n]
            # Calculate scaling values for bboxes.
            scale = input_image.shape[2]
            scale /= self.scale
            scale_y = scale
            scale_x = scale
            if self.nms_threshold and scores[n] < self.nms_threshold:
                continue
            # Append to detections
            detected_objects.append({
                'ymin': boxes[0][n][0] * scale_y,
                'xmin': boxes[0][n][1] * scale_x,
                'ymax': boxes[0][n][2] * scale_y,
                'xmax': boxes[0][n][3] * scale_x,
                'score': scores[0][n],
                'class': int(pred_classes[0][n]),
                'mask': mask,
            })

        print(detected_objects)
        return detected_objects


    # def _visualize(self, detected_objects, input_image):
    #     """
    #     Composes the processed results onto the input image to create visual 
    #     representation of the detections
    
    #     NOTE: for development purposes only

    #     :params:
    #         - detected_objects: processed inference results
    #         - input_image: original image input 

    #     :returns:
    #         - the input image with the detecitons visualized 
    #     """
    #     input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    #     image = Image.fromarray(input_image_rgb)
    #     # Get image dimensions.
    #     im_width, im_height = image.size
    #     line_width = 2
    #     font = ImageFont.load_default()
    #     for d in detected_objects:
    #         color = COLORS[d['class'] % len(COLORS)]
    #         # Dynamically convert PIL color into RGB numpy array.
    #         pixel_color = Image.new("RGB",(1, 1), color)
    #         # Normalize.
    #         np_color = (np.asarray(pixel_color)[0][0])/255
    #         # TRT instance segmentation masks.
    #         if isinstance(d['mask'], np.ndarray) and d['mask'].shape == (28, 28):
    #             # PyTorch uses [x1,y1,x2,y2] format instead of regular [y1,x1,y2,x2].
    #             d['ymin'], d['xmin'], d['ymax'], d['xmax'] = d['xmin'], d['ymin'], d['xmax'], d['ymax']
    #             # Get detection bbox resolution.
    #             det_width = round(d['xmax'] - d['xmin'])
    #             det_height = round(d['ymax'] - d['ymin'])
    #             # Slight scaling, to get binary masks after float32 -> uint8
    #             # conversion, if not scaled all pixels are zero.
    #             mask = d['mask'] > self.iou_threshold
    #             # Convert float32 -> uint8.
    #             mask = mask.astype(np.uint8)
    #             # Create an image out of predicted mask array.
    #             small_mask = Image.fromarray(mask)
    #             # Upsample mask to detection bbox's size.
    #             mask = small_mask.resize((det_width, det_height), resample=Image.BILINEAR)
    #             # Create an original image sized template for correct mask placement.
    #             pad = Image.new("L", (im_width, im_height))
    #             # Place your mask according to detection bbox placement.
    #             pad.paste(mask, (round(d['xmin']), (round(d['ymin']))))
    #             # Reconvert mask into numpy array for evaluation.
    #             padded_mask = np.array(pad)
    #             #Creat np.array from original image, copy in order to modify.
    #             image_copy = np.asarray(image).copy()
    #             # Image with overlaid mask.
    #             masked_image = self.__overlay(image_copy, padded_mask, np_color)
    #             # Reconvert back to PIL.
    #             image = Image.fromarray(masked_image)

    #         # Bbox lines.
    #         draw = ImageDraw.Draw(image)
    #         draw.line([(d['xmin'], d['ymin']), (d['xmin'], d['ymax']), (d['xmax'], d['ymax']), (d['xmax'], d['ymin']),
    #                 (d['xmin'], d['ymin'])], width=line_width, fill=color)
    #         label = "Class {}".format(d['class'])
    #         if d['class'] < len(LABELS):
    #             label = "{}".format(LABELS[d['class']])
    #         score = d['score']
    #         text = "{}: {}%".format(label, int(100 * score))
    #         if score < 0:
    #             text = label
    #         text_width, text_height = font.getsize(text)
    #         text_bottom = max(text_height, d['ymin'])
    #         text_left = d['xmin']
    #         margin = np.ceil(0.05 * text_height)
    #         draw.rectangle([(text_left, text_bottom - text_height - 2 * margin), (text_left + text_width, text_bottom)],
    #                     fill=color)
    #         draw.text((text_left + margin, text_bottom - text_height - margin), text, fill='black', font=font)

    #         return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # def __overlay(self, image, mask, color, alpha_transparency=0.5):
    #     """
    #     Overlay mask with transparency on top of the image.
        
    #     :params:
    #         - image: a copy of original image in RGB pil format
    #         - mask: the mask (segmentation) to overlay on the image
    #         - color: the color of the visualized mask
    #         - alpha_transparency: transparency of the mask, default 0.5
        
    #     :returns:
    #         - the image with the mask overlayed
    #     """
    #     for channel in range(3):
    #         image[:, :, channel] = np.where(
    #             mask == 1,
    #             image[:, :, channel] *
    #             (1 - alpha_transparency) + alpha_transparency * color[channel] * 255,
    #             image[:, :, channel]
    #             )
    #     return image

