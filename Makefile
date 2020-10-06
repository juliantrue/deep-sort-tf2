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

# Train with the best hyperparameters
.PHONY: train
train:
	python3 deepsort/deep/train.py

# Test the result of the best hyperparameters
.PHONY: test
test:
	python3 deepsort/deep/test.py

# Run the full MOT16 tracking suite on the resulting program
.PHONY: eval
eval:
	python3 evaluate.py 
	python3 -u -m motmetrics.apps.eval_motchallenge /MOT16/train ./results/

# Perform a grid search for the best hyperparameters
.PHONY: hparam_sweep
hparam_sweep:
	python3 deepsort/deep/train.py --train_mode hyperparameter

# Fine tune with the hessian penalty
.PHONY: fine_tune
fine_tune:
	python3 deepsort/deep/fine_tune.py 

.PHONY: clean
clean:
	sudo rm -r ./logs/*
