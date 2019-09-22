FROM wsi/cuda90:latest

# Run as root
USER root

# Create a working directory
RUN mkdir /digestpath
COPY src /digestpath/
WORKDIR /digestpath
