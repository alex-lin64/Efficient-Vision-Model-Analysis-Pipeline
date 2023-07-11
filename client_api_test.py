from client.client_api import BaselineClient


if __name__ == '__main__':
    yolov7_client = BaselineClient(model='yolov7')
    # yolov7_client.inference(input_='', mode='dummy')
    # yolov7_client.inference(input_='data/messy_kitch.jpg', output_='data/messy_kitch_result.jpg', mode='image')
    # yolov7_client.inference(input_='data/1.mp4', mode='video')
    
