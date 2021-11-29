Movie reviews sentiment analysis
==============================
Overview
------------
This project was executed as a part of the [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) course held by [DataTalks.Club](https://datatalks.club/). 
It covers binary classification problem, but at the same time has some features related to text data processing. 
It doesn't contain complex approaches and uses only standard libraries of 'classical' machine learning. 
The project covers the main stages of development according to the CRISP-DM methodology.

Problem description
------------
Midterm project is related to movie reviews sentiment analysis based on IMDB Database.
The main goal is to correctly classify the positive and negative sentiments for IMDB reviews. 
The task itself is very important, especially in such business areas as marketing and ads. 
Sentiment analysis can help you determine your customers' level of satisfaction based on their feedback.
Or even detect scam, abuse or trolling in Twitter or forum posts. That's why i decided to pay attention to this topic.

Dataset
------------
The original data is represented by [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) for natural language processing. 
It’s a dataset containing movie reviews (text) for sentiment analysis (binary — positive or negative).
The dataset contains two columns - review and sentiment to perform the sentimental analysis.
More details can be found [here](http://ai.stanford.edu/~amaas/data/sentiment/).

Project structure
------------
The project structure is similar to [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project template.

    ├── configs                     <- Configuration files for projects modules.
    │   └── train_config.yaml       <- Training pipeline configuration file.
    │
    ├── data
    │   ├── external                <- Data from third party sources.
    │   ├── processed               <- The final, canonical data sets for modeling.
    │   └── raw                     <- The original, immutable data dump.
    │
    ├── models                      <- Trained and serialized models and pipelines.
    │
    ├── notebooks                   <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                  the creator's initials, and a short `-` delimited description, e.g.
    │                                  `1.0-jqp-initial-data-exploration`.
    │
    ├── src                         <- Source code for use in this project.
    │   │
    │   ├── data                    <- Scripts to download and preprocess data.
    │   │   └── process_data.py
    │   │
    │   ├── entities                <- Simple classes for input/output data.
    │   │   └── movie_reviews.py
    │   │
    │   ├── models                  <- Scripts to train models and then use trained models to make
    │   │   │                          predictions.
    │   │   └── model_fit_predict.py
    │   │
    │   ├── send_request.py
    │   └── train_pipeline.py
    │
    ├── predict.py                  <- FastAPI application code.
    │
    ├── Dockerfile                  <- Docker image building file.
    │
    ├── README.md                   <- The top-level README for developers using this project.
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment.
    │
    └── requirements-deploy.txt     <- The requirements file for the service deployment.

Tutorial
------------
### How to start?
I believe the best way to get started is to read [Jupyter notebooks](notebooks) that contain data preprocessing, exploratory data analysis (EDA in short), model training and validation, etc.
### Virtual environment setup
If you want to reproduce the model training results by running [notebooks](notebooks) or [`train_pipeline.py`](src/train_pipeline.py), then you need to create a virtual environment and install the dependencies.
I used `conda` package manager from [Anaconda distribution](https://www.anaconda.com/) for doing that, but you feel free to choose any other tools (`pipenv`, `venv`, etc.).
In case of `conda`, just follow the steps below:
1. Open the terminal and choose the project directory.
2. Create new virtual environment by command `conda create -n test-env python=3.8`.
3. Activate this virtual environment with `conda activate test-env`.
4. Install all package using `pip install -r requirements.txt`.

And that's all! Now you can easily run the scripts and notebooks.
### Run service
To run the service locally in your environment, simply use the following command:
```bash
uvicorn predict:app --host=0.0.0.0 --port=8000
```
### Containerization
Here is a short guide how to build an image from Dockerfile and run then a container on your local machine.
Be sure that you have already installed the Docker, and it's running on your machine now.
1. Open the terminal and choose the project directory.
2. Build docker image from [`Dockerfile`](Dockerfile) using `docker build -t movie-reviews:latest .`.
With `-t` parameter we're specifying the tag name of our docker image. 
3. Now use `docker run -it -p 8000:8000 movie-reviews:latest` command to launch the docker container with your app. 
Parameter `-p` is used to map the docker container port to our real machine port.

Instead of building an image by yourself, you can pull it out an already built from [Dockerhub](https://hub.docker.com/). 
Use the command `docker pull svizor/movie-reviews:latest` in this case.
### Heroku docker deployment
This part contains the steps of deploying your app to Heroku platform. Just follow it:
1. Register on [Heroku](https://signup.heroku.com/) and install Heroku CLI.
2. After CLI installation, open the terminal and run the `heroku login` command to login to Heroku.
3. At the same terminal login to Heroku container registry using `heroku container:login` command.
4. Then create a new app in Heroku with the following command `heroku create movie-reviews-docker`.
5. Make small changes in [`Dockerfile`](Dockerfile): uncomment the last line and comment out the line above. 
The only difference between these two lines is a port number. Heroku automatically assigns `$PORT` number from the dynamic pool. 
So, there is no need to specify it manually.
6. Next, run the `heroku container:push web -a movie-reviews-docker` command to push docker image to Heroku.
7. Release the container using the command `heroku container:release web -a movie-reviews-docker`.
8. Launch your app by clicking on generated URL. In our case it should be https://movie-reviews-docker.herokuapp.com/. 
If we have successfully deployed the app, the link opens without problems.

Now we can move on to the next step - service testing.
### Service testing
There are a couple of ways to test the deployed service:
* use the Swagger UI provided by FastAPI framework (https://movie-reviews-docker.herokuapp.com/docs in our case);
* or use the provided script [send_request.py](src/send_request.py) that extracts data from a file and sends requests to the service https://movie-reviews-docker.herokuapp.com/predict endpoint.

In case of using a script, just follow the command format:
```python
python src/make_request.py --host=<host_address> --dataset_path=<file_path>
```
To test our Heroku deployment, we should type:
```python
python src/send_request.py --host=movie-reviews-docker.herokuapp.com
```