docker run -dit --privileged  --gpus 'all' --entrypoint bash --shm-size 1G  --name=triton_xyz_1 -v /es01:/es01 -v /home:/host_home triton_baichuan:23.10.1