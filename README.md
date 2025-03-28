编译容器启动命令：
docker run -it --gpus all \
  --privileged \
  -v /data:/data:rw \
  -p 18777:18777 -p 5555:22 \
  --ipc=host \
  --name sanhe_grpc_compiler \
  ab993234d560 /bin/bash
  