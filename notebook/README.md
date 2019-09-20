# Docker Usage
## Build image
```
$ docker build -t digestpath:chenpingjun_task2 .
```

## Save image
```
$ docker save digestpath:chenpingjun_task2 > chenpingjun_task2.tar
```

## Load image
```
$ docker load --input chenpingjun_task2.tar
```

## Show available images
```
$ docker images
```

## Start container
```
$ docker run -it --name chenpingjun --restart always -v ~/CPath/colon-tissue-seg:/App digestpath:chenpingjun
$ docker run -dit --runtime nvidia  --restart always --shm-size 60G --name chenpingjun -v /home/pingjun/CPath/colon-tissue-seg/DigestPath/data/input:/input:ro -v /output -e NVIDIA_VISIBLE_DEVICES=5 digestpath:chenpingjun_task2
$ docker exec -u 0 -it chenpingjun /bin/bash
$ docker exec -it chenpingjun python /digestpath/Segmentation.py
$ docker exec -it chenpingjun /bin/bash
```

## Show running containers
```
$ docker ps
```

## Stop container
```
$ docker stop chenpingjun
$ docker rm chenpingjun
```
