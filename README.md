Tasks:
    -- visualization 
    -- cvat integration
    -- data storage
    -- connect from separate machine
    -- organize code/comments
    -- client for detectron2 


    Nice to haves:
        ** batching the inputs if input is directory?  how to?

        ** auto detect video or image in a file?  might need magic package



docker run --gpus all --rm --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 --mount type=bind,source="/mnt/c/Users/Alex Lin/Desktop/baseline_system/triton_deploy/models",destination=/models nvcr.io/nvidia/tritonserver:22.06-py3 tritonserver --model-repository=/models --strict-model-config=false --log-verbose 1


weird issue with conda envs and pip:

    run in conda env once after creating the env:
        python -m pip install --upgrade --force-reinstall pip


detectron2 - inputs must be 1344x1344, due to tensorrt not having dynamic shape plugins


cvat server username: django
pass: bfc


triton base client to test:

    One all client inferences implemented
        -- inference image only vis
        -- inference image only export to file
        -- inference image only export to fo
        -- inference image export file and fo and vis
        -- inference video only vis
        -- inference video only export to file
        -- inference video only export to fo




