pwd=${PWD}
project_name=cosine-metric-learning
.PHONY: all
all: build run

.PHONY: build
build:
	sudo docker build -t ${project_name}:latest .

.PHONY: run
run: 
	sudo docker run --gpus all --rm --network=host -it -v ${pwd}:/${project_name} ${project_name}:latest bash

.PHONY: clean
clean:
	sudo rm -r ./logs/*
	sudo rm -r ./checkpoints/c*
