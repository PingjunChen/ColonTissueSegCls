# Docker Usage
## Build image
```
$ docker build -t digestpath:chenpingjun .
```

## Save image
```
$ docker save digestpath:chenpingjun > chenpingjun_task2.tar
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
$ docker run -dit --runtime nvidia  --shm-size=16G --restart always --name chenpingjun -v /home/pingjun/CPath/colon-tissue-seg/DigestPath/data/input:/input:ro -v /output digestpath:chenpingjun
$ cd
$ docker exec -u 0 -it chenpingjun /bin/bash
$ docker exec -it chenpingjun python /digestpath/Segmentation.py
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
