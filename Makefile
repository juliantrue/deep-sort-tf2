pwd=${PWD}
project_name=deep-sort-tf2
MOT16_PATH=/home/julian/Datasets/MOT16

.PHONY: all
all: build run

.PHONY: build
build:
	sudo docker build -t ${project_name}:latest .

.PHONY: run
run: 
	xhost + 
	sudo docker run --gpus all \
		--rm --network=host -it \
		-e DISPLAY=${DISPLAY} \
		-v /tmp/.X11-unix/:/tmp/.X11-unix \
		-v ${MOT16_PATH}:/MOT16 \
		-v ${pwd}:/${project_name} \
		${project_name}:latest bash

.PHONY: train
train:
	python3 deepsort/deep/train.py

.PHONY: test
test:
	python3 deepsort/deep/test.py

.PHONY: eval
eval:
	python3 evaluate.py 
	python3 -u -m motmetrics.apps.eval_motchallenge /MOT16/train ./results/


.PHONY: clean
clean:
	sudo rm -r ./logs/*
	sudo rm -r ./checkpoints/c*
