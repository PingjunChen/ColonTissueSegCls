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
$ docker run -it --name ColonSeg --restart always -v ~/CPath/colon-tissue-seg:/App colon
$ docker run -dit --name chenpingjun --restart always -v /home/pingjun/CPath/colon-tissue-seg/DigestPath/data/input:/input:ro -v /output digestpath:chenpingjun
$ docker run -dit --runtime nvidia  --shm-size=256g --name chenpingjun --restart always -v /home/pingjun/CPath/colon-tissue-seg/DigestPath/data/input:/input:ro -v /output digestpath:chenpingjun
$ docker exec -it chenpingjun /bin/bash
$ docker exec -it chenpingjun python /digestpath/Segmentation.py

```

## Show running containers
```
$ docker ps
```

## Stop container
```
$ docker stop ColonSeg
$ docker rm ColonSeg
```
