FROM tensorflow/tensorflow:latest-gpu

RUN pip3 --no-cache-dir install \
      tqdm \
      tensorflow_datasets




