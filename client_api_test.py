from client.yolov7_client.yolov7_triton_client import YoloV7_Triton_Inference_Client
from client.detectron2_client.detectron2_triton_client import Detectron2_Triton_Client


if __name__ == '__main__':
    url = '192.168.1.58:8001'
    fo_dataset = "test_run1"
    yolov7_client = YoloV7_Triton_Inference_Client(url=url)
    detectron2_client = Detectron2_Triton_Client(width=1344, height=1344)
    # yolov7_client.infer_dummy()
    yolov7_client.infer_image(input_='data/raw/images', vis=True)
    # yolov7_client.infer_video(input_='data/raw/video', output_='data/inference/video')

    # detectron2_client.infer_dummy()
    # detectron2_client.infer_image(input_='data/raw/images/test')
    # detectron2_client.infer_video(input_='data/raw/video', output_='data/inference/video')
    
