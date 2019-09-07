# Docker Usage
## Build image
```
$ docker build -t colon docker/
```

## Show available images
```
$ docker images
```

## Start container
```
$ docker run -it --name ColonSeg --restart always colon
$ docker run -it --name ColonSeg --restart always -v ~/CPath/colon-tissue-seg:/App colon
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
