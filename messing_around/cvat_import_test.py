from time import sleep
from cvat_sdk.api_client import Configuration, ApiClient, exceptions
from cvat_sdk.core.proxies.tasks import ResourceType, Task
from cvat_sdk.api_client.models import *


configuration = Configuration(
    host = "http://localhost:8080",
    username = 'django',
    password = 'bfc'
)

with ApiClient(configuration) as client:
    # task id
    id = 1 
    # finishes data upload
    upload_finish = True 
    # indicates data is single or multiple files attached to a task
    upload_multiple = True 
    # initializes data upload
    upload_start = True 
    # task specs for a new task
    task_spec = {
        'name': 'test tasks',
        "labels": [{
            "name": "car",
            "color": "#ff00ff",
            "attributes": [
                {
                    "name": "a",
                    "mutable": True,
                    "input_type": "number",
                    "default_value": "5",
                    "values": ["4", "5", "6"]
                }
            ]
        }],
    }
    # creat a new task
    data_request = DataRequest(
        image_quality=100,
        client_files=[
            open('C:\\Users\Alex Lin\\Desktop\\baseline_system\\data\\raw\\images\\dog.jpg', 'rb'), 
            open('C:\\Users\\Alex Lin\\Desktop\\baseline_system\\data\\raw\\images\\messy_kitch.jpg', 'rb'),
            open('C:\\Users\\Alex Lin\\Desktop\\baseline_system\\data\\raw\\images\\test\\horse.png', 'rb')
            ],
    ) 

    try:
        # Apis can be accessed as ApiClient class members
        # We use different models for input and output data. For input data,
        # models are typically called like "*Request". Output data models have
        # no suffix.
        (task, response) = client.tasks_api.create(task_spec)
    except exceptions.ApiException as e:
        # We can catch the basic exception type, or a derived type
        print("Exception when trying to create a task: %s\n" % e)

    
    (_, response) = client.tasks_api.create_data(
        task.id,
        upload_finish=upload_finish,
        upload_multiple=upload_multiple,
        upload_start=upload_start,
        data_request=data_request,
        _content_type="multipart/form-data"
    )

    assert response.status == 202, response.msg

    # Wait till task data is processed
    for _ in range(100):
        (status, _) = client.tasks_api.retrieve_status(task.id)
        if status.state.value in ['Finished', 'Failed']:
            break
        sleep(0.1)
    assert status.state.value == 'Finished', status.message

    # Update the task object and check the task size
    (task, _) = client.tasks_api.retrieve(task.id)
    assert task.size == 3
    


    
    



    