# Docker Usage
## Image management
```
$ docker build -t digestpath:chenpingjun_task2 .                    # build the image
$ docker save digestpath:chenpingjun_task2 > chenpingjun_task2.tar  # save image
$ docker load --input chenpingjun_task2.tar                         # load image
$ docker images                                                     # Show available images
```

## Start container
```
$ docker run -dit --runtime nvidia  --restart always --shm-size 16G --name chenpingjun -v /home/pingjun/CPath/colon-tissue-seg/DigestPath/data/input:/input:ro -v /output -e NVIDIA_VISIBLE_DEVICES=5 digestpath:chenpingjun_task2
$ docker exec -it chenpingjun python /digestpath/Segmentation.py
$ docker exec -u 0 -it chenpingjun /bin/bash
```

## Container management
```
$ docker ps                # list running containers
$ docker stop chenpingjun  # stop running container with name chenpingjun
$ docker rm chenpingjun    # remove container with name chenpingjun
```

## Copy files
```
$ docker cp chenpingjun:/output .
```
