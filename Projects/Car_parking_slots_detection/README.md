Car parking slots detection
==============================
Overview
------------
This project was completed as part of the [Machine Learning Zoomcamp course](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) held by [Alexey Grigorev](https://de.linkedin.com/in/agrigorev) from [DataTalks.Club](https://datatalks.club/).
It covers the field of deep learning - the problem of classification, to be precise. For more details, please, go through the paragraphs below.

Problem description
------------
Finding a free space in a parking zone is a rather hard problem. It is more difficult task to manage such objects with varying levels of incoming/outcoming traffic. Which parking lots are vacant right now? Or when do we need to add some more slots?

There are several approaches exist that solve this problem. The most popular is to mount a special sensors on the ground at each parking slot. Sensors are quite effective, but they need a maintenance and too expensive. 

Another approach is to use the camera systems (CCTV) that are pre-installed for security purposes. These cameras could help to detect parking slots occupancy using deep learning algorithms. 
Deep learning models are used to determine the occupancy of parking spaces from images obtained by cameras. 
Such deep neural networks achieve high accuracy and applicability in real-time environments. Compared to the sensor-based methods, there is no need to install expensive sensors for every parking space.
AI vision-based systems are also highly scalable and can be used in indoor environments like mega shopping malls or outdoor parking lots.

Dataset
------------
The initial dataset is called [CNRPark+EXT - A Dataset for Visual Occupancy Detection of Parking Lots](http://cnrpark.it/). This is a dataset for visual occupancy detection of parking lots of roughly 150,000 labeled images (patches) of vacant and occupied parking spaces, built on a parking lot of 164 parking spaces.

Project structure
------------
The project structure is similar to [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project template.

    ├── configs                         <- Configuration files for projects modules.
    │   └── train_config.yaml           <- Training pipeline configuration file.
    │
    ├── data
    │   ├── external                    <- Data from third party sources.
    │   ├── processed                   <- The final, canonical data sets for modeling.
    │   └── raw                         <- The original, immutable data dump.
    │
    ├── logs                            <- Logs generated during the models training (for tensorboard).
    │
    ├── models                          <- Trained and serialized models and pipelines.
    │
    ├── notebooks                       <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                   the creator's initials, and a short `-` delimited description, e.g.
    │                                   `1.0-jqp-initial-data-exploration`.
    │
    ├── src                             <- Source code for use in this project.
    │   │
    │   ├── data                        <- Scripts to download and preprocess data.
    │   │   └── process_data.py
    │   │
    │   ├── entities                    <- Simple classes for input/output data.
    │   │   ├── image_params.py
    │   │   ├── parking_slot_response.py
    │   │   ├── movie_reviews.py
    │   │   └── movie_reviews.py
    │   │
    │   ├── models                      <- Scripts to train models and then use trained models to make
    │   │   │                           predictions.
    │   │   └── model_fit_predict.py
    │   │
    │   ├── convert.py
    │   ├── send_request.py
    │   └── train_pipeline.py
    │
    ├── predict.py                      <- FastAPI application code.
    │
    ├── Dockerfile                      <- Docker image building file.
    │
    ├── README.md                       <- The top-level README for developers using this project.
    │
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment.
    │
    └── requirements-deploy.txt         <- The requirements file for the service deployment.

Tutorial
------------
### How to start?
The best way to get started is to read [Jupyter notebooks](notebooks) that contain data preparations, exploratory data analysis (EDA in short), step by step model training and validation, etc.
### Reproducibility
The reproducibility in deep learning in really tricky question. I tried to fix most of the variables as much as possible, but it is quite difficult to exactly reproduce the output data.
However, the results should be close to those that provided in the notebooks. 

All the results are produced on OS Windows 10.
### Virtual environment setup
If you want to reproduce the model training results by running [notebooks](notebooks) or [`train_pipeline.py`](src/train_pipeline.py), then you need to create a virtual environment and install the dependencies.
I used `conda` package manager from [Anaconda distribution](https://www.anaconda.com/) for doing that, but you feel free to choose any other tools (`pipenv`, `venv`, etc.).
In case of `conda`, just follow the steps below:
1. Open the terminal and choose the project directory.
2. Create new virtual environment by command `conda create -n test-env python=3.9`.
3. Activate this virtual environment with `conda activate test-env`.
4. Install all package using `pip install -r requirements.txt`.

And that's all! Now you can easily run the scripts and notebooks.
### Run scripts
You can find 2 scripts in the [`src`](src) directory:
* [`train_pipeline`](src/train_pipeline.py) - the main script that implements model training and produces keras model in `.h5` format on the output;
* [`convert.py`](src/convert.py) - a small script for converting model to TensorFlow Lite format `.tflite`.

So, you should start with train pipeline first and then convert the model to `.tflite` format that which is used by the ML service for online inference.

__Note:__ If you are get the error message `ModuleNotFoundError: No module named 'src'` when running the training script, try to execute the following command in the terminal on Windows:
 ```
set PYTHONPATH=%PYTHONPATH%;\path\to\project
```
On Linux, please, use `export` command instead of `set`.
### Run service
Two important things you should check before running ML service locally:
* set up environmental variable to the trained model location, e.g. `set MODEL_PATH=models/model.tflite`;
* make sure that the code line `import tensorflow.lite as tflite` is uncomment while `import tflite_runtime.interpreter as tflite` is commented.

When you decide to deploy the model, you should do the opposite actions.

To run the service locally in your environment, simply use the following command:
```bash
uvicorn predict:app --host=0.0.0.0 --port=8000
```
### Containerization
Here is a short guide how to build an image from Dockerfile and run then a container on your local machine.
Be sure that you have already installed the Docker, and it's running on your machine now.
1. Open the terminal and choose the project directory.
2. Build docker image from [`Dockerfile`](Dockerfile) using `docker build -t parking-slots:latest .`.
With `-t` parameter we're specifying the tag name of our docker image. 
3. Now use `docker run -it -p 8000:8000 parking-slots:latest` command to launch the docker container with your app. 
Parameter `-p` is used to map the docker container port to our real machine port.

Instead of building an image by yourself, you can pull it out an already built from [Dockerhub](https://hub.docker.com/). 
Use the command `docker pull svizor/parking-slots:latest` in this case.
### Heroku docker deployment
This part contains the steps of deploying your app to Heroku platform. Just follow it:
1. Register on [Heroku](https://signup.heroku.com/) and install Heroku CLI.
2. After CLI installation, open the terminal and run the `heroku login` command to login to Heroku.
3. At the same terminal login to Heroku container registry using `heroku container:login` command.
4. Then create a new app in Heroku with the following command `heroku create parking-slots-docker`.
5. Make small changes in [`Dockerfile`](Dockerfile): uncomment the last line and comment out the line above. 
The only difference between these two lines is a port number. Heroku automatically assigns `$PORT` number from the dynamic pool. 
So, there is no need to specify it manually.
6. Next, run the `heroku container:push web -a parking-slots-docker` command to push docker image to Heroku.
7. Release the container using the command `heroku container:release web -a parking-slots-docker`.
8. Launch your app by clicking on generated URL. In our case it should be https://parking-slots-docker.herokuapp.com/. 
If we have successfully deployed the app, the link opens without problems.

Now we can move on to the next step - service testing.
### Service testing
You can check two endpoints to be sure that everything is good with the deployed service:
* [healthcheck](https://parking-slots-docker.herokuapp.com/health) shows that the model was successfully loaded and now is used by the service;
* [prediction endpoint](https://parking-slots-docker.herokuapp.com/predict) serves for the model scoring.

To test the prediction endpoint you can use handmade script [send_request.py](src/send_request.py) that takes data from a specified directory and sends requests to the service.

In case of using a script, just follow the command format:
```python
python src/send_request.py --host=<host_address> --dataset_path=<data_dir_path>
```
To test our Heroku deployment, we should type:
```python
python src/send_request.py --host=parking-slots-docker.herokuapp.com
```
You can also use this script to test the service that is running locally.

To-Do list
------------
- [ ] Finetune other pretrained models
- [ ] Make predictions for the whole image from the camera
- [ ] Use detection / segmentation approach
- [ ] Write unit tests