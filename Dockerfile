FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get -y  --no-install-recommends install \
    libgl1-mesa-glx \
    qt5-default

RUN pip3 --no-cache-dir install \
      tqdm \
      opencv-python \ 
      tensorflow_datasets




