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
	sudo docker run --gpus all \
		--rm --network=host -it \
		-v ${MOT16_PATH}:/MOT16 \
		-v ${pwd}:/${project_name} \
		${project_name}:latest bash

.PHONY: clean
clean:
	sudo rm -r ./logs/*
	sudo rm -r ./checkpoints/c*
