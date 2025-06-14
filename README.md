# Student Performance ML Pipeline

**Capstone Project — MLOps Zoomcamp 2025**

This project is an end-to-end ML workflow that predicts a student's final grade class (A–F) using demographic and academic performance data. It uses XGBoost as the classifier, and Prefect + MLflow for orchestration and experiment tracking. In addition, I use Evidently to track potential data drift.

For ease of use, you can interact with the deployed Streamlit app on [Huggingface Spaces](https://huggingface.co/spaces/selbl/gradeclass-prediction). This is a minimalistic version of the app that provides a simple yet effective user interface to perform inference from the best performing model.

## Note to Evaluators

For ease of evaluation, I provide some comments here addressing all of the required criteria:

- **Problem description**: Please see above. My project's objective is to estimate a student's predicted grade on a class given information of their academic performance, parental situation, ethnicity, etc... If I were to train this with a sufficiently large sample it could provide guidance to regulators on which activities/behaviours to focus more in order to increase the chances of students getting good grades (such as targetting those who are not enrolled in extracurriculars and encouraging them to do so, etc...)
- **Cloud**: I have deployed my model to DockerHub (via the CD pipeline) as well as having a Streamlit app that runs my model
- **Experiment Tracking and Model Registry**: I use both experiment tracking and model registry by using MLFlow. I only include the final experiment in this repo in order to prevent bloat. You can check the files prefect_train_xgb.py and prefect_register_model.py to check its implementation
- **Workflow Orchestration**: I use Prefect to orchestrate my model. You can check prefect_orchestrate_pipeline.py for details
- **Model Deployment**: My code is containerized in a Docker container and can be deployed to the cloud (you can try with an EC2 instance for example)
- **Model Monitoring**: I use Evidently to track the metrics for data drift in order to ensure that the train and validation sets have similar distributions
- **Reproducibility**: I try to be very detailed with my instructions, so hopefully this works!
- **Best Practices**: I have included all of the suggested best practices. Please check the final section of this README for details


## Project Structure

```
mlops-zoomcamp-project-25/
│
├── code/                       ← all source code & pipelines
│   ├── prefect_process_data.py
│   ├── prefect_train_xgb.py
│   ├── prefect_register_model.py
│   ├── prefect_test.py
│   ├── flask_predict.py
│   ├── flask_test.py
│   ├── prefect_orchestrate_pipeline.py
│   ├── mlflow.db                        ← tracked
│   └── mlartifacts/
│       └── 2/ …                         ← tracked
│
├── data/                       ← all data
│   ├── students_performance.csv        ← original data source
│   └── processed/, raw/                 # created dynamically
│
├── tests/ 
│
├── Dockerfile
├── Pipfile / Pipfile.lock
└── .gitignore
```

This repository only has the final run artifacts in `code/mlartifacts/2` and `mlflow.db`. I omit the rest because of space constraints.

I did mess up some of the configuration because I use the mlflow.db database from within the code folder. My instructions try to specify on which folder you should be at all times.

---

## Tech Stack

* **Model**: XGBoost
* **Orchestration**: Prefect
* **Monitoring**: Evidently
* **Experiment Tracking**: MLflow Tracking + Model Registry
* **Serving**: Flask + Gunicorn, Streamlit (on Hugginface Spaces)
* **Dev Tools**: Docker, Pipenv

---

## Data

I use the [Student Performance Dataset from Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset). This rich dataset has information on high school students demographics (age, ethnicity) and academic performance (absences, extracurriculars) as well as their final grade.

I decided to go with a relatively simple and clean dataset in order to fully focus on the deployment part of the process. I use XGBoost as my predictive model in order to leverage the power of gradient boosted decision trees on a relatively small sample. This allows me to achieve high performance while having a relatively low training cost.

---

## Quickstart (Online)

The simplest way to interact with the app is via the [Huggingface Spaces](https://huggingface.co/spaces/selbl/gradeclass-prediction) Streamlit deployed app. This provides a easy way to perform inference with a simple UI and without needing to install anything in your computer.

Though very handy, the Huggingface spaces app does not provide much insights on all of the work behind the scenes I did for this project. Please check the local quickstart guide for a step by step approach to working with the model locally and to check the whole orchestration pipeline.

## Quickstart (Local)

The instructions below show how to interact with the app locally in case you want to take a look at all of the orchestration and step by step approach. I suggest using Docker (Option B) to make your life easier, though you can also look at each of the separate components of the code as well.

### Clone repo

The first step is to clone the repo:

```bash
git clone "https://github.com/Selbl/mlops-zoomcamp-project-2025"
```

### Install dependencies

From within the main folder (mlops-zoomcamp-project-25)

```bash
pip install pipenv
pipenv shell
```

With this you have the necessary environment to run the code. Below, I describe how to run the whole end-to-end process which includes training and model registry (Option A). Alternatively, you can run the Docker image (Option B) which allows you to quickstart your inference journey.

Note that the end-game of the model is to performance inference on one observation of a hypothetical student. In this version of my workflow, you have to manually edit either the test case in flask_test.py or the sample .csv file in data/sample_test_input.csv which you then run using prefect_test.py. The steps below detail how you would go about working with the whole pipeline.

### Option A. Step by step process

#### A.1. Launch Prefect + MLflow UI

From within the main folder you should change to the code/ folder (first cd command in this example) and then run the rest.

```bash
cd code
prefect server start
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

#### A.2. Run end-to-end pipeline

The whole pipeline consists of:


The whole pipeline consists of:

1 - Data pre-processing (prefect_process_data.py)
2 - Data drift monitoring using Evidently (evidently_data_drift_check.py)
3 - Training routine with MLFlow (prefect_train_xgb.py)
4 - Model registering with MLFlow (prefect_register_model.py)

In order to make the model run faster in your machine, the hyperparameter search grid in the code from the repo is quite small. You can uncomment the more detailed grid if you want to, but it takes a while to run.

Use the process flag if you want to process the data by yourself (generate the train, validation, test split).

```bash
python prefect_orchestrate_pipeline.py --process True
```

#### A.3. Serve latest model with Flask

Be sure to be inside code/ when running this command. The Flask server runs on http://localhost:9696

```bash
python flask_predict.py
```

#### A.4. Test prediction

Conduct a prediction on the JSON object inside of flask_test.py

```bash
python flask_test.py
```

Alternatively, you can use the prefect_test.py file to run a test on the sample test file located in ../data/sample_test_input.csv (the path is relative to the code/ folder)

```bash
python prefect_test.py
```

---

### B. Dockerized Inference

#### B.1 Build image

Remember that you have to be inside the main folder (mlops-zoomcamp-project-25) in order to build the image:

```bash
docker build -t gradeclass-service .
```

#### B.2 Run container

Run the following to run the container:

```bash
docker run -p 9696:9696 -p 5001:5000 -p 4200:4200 \
          -v "$(pwd)/code:/app/code" \
          gradeclass-service
```

NOTE: I am using 5001:5000 for MLFlow because on my machine it is sometimes used by another process (5001:5000 refers to 5001 in the host machine and 5000 in the container). Feel free to edit it to 5000:5000 if you wish, but it should not change a thing.

This launches:

* REST API on [http://localhost:9696](http://localhost:9696)
* MLflow UI inside the container on [http://localhost:5000](http://localhost:5000)
* Prefect inside the container (usually) on [http://localhost:4200](http://localhost:4200)

Note that you can add -p 5000:5000 to the code above in case you want to have access to the MLFlow UI.

I do not provide the Evidently UI to prevent bloat and because my data is quite static. Instead, you can check whether you had datadrift by looking at the logs of the prefect run or the print statements from the orchestration code.

#### B.3 Inference

You can then open a new terminal and run the following in order to perform inference:

```bash
python code/flask_test.py
```

---

## Better Practices

My code includes:

- **Unit tests**: Checks on the data processing functions as well as the existence of the necessary data for the code to run
- **Integration tests**: Checking that the Docker image is healthy and connects to the desired endpoint
- **Linter**: I use lint to format my code before committing
- **Makefile**: For linting, testing and running the Docker image
- **Pre-commit hooks**: Automatically checks for code formatting and potential errors before committing 
- **CI/CD**: I implement a CI/CD pipeline where every push and pull request to main triggers those events. The CI does some checks to ensure everything with the Docker image is in order and the CD deploys the image to [Dockerhub](https://hub.docker.com/repository/docker/selbl/gradeclass-service/general)

