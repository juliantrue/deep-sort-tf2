THIS IS STILL A WORK IN PROGRESS. USE AT YOUR OWN RISK

# Setup
This project will only run on a linux based OS.

## Dependencies
Install the latest version of Docker to run this project.

# Instructions
To run the environment, simply type `make` into the terminal. Once you are in the 
docker container, you may run your job.

## Training:
Simply run `make train` to train the custom model on the included MARS dataset. 
`make test` to test the trained model.


## MOT16 Evaluation
Download the [MOT16 dataset](https://motchallenge.net/data/MOT16.zip). Unzip this dataset.
In the Makefile, change the variable `MOT16_PATH` to the location where MOT16 dataset is stored.


Once this is all done, `make eval` will evaluate the model and Deep SORT on MOT16.



