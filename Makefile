.PHONY: format lint test run docker-build

format:
	black .

lint:
	flake8 .

test:
	pytest             

itest:
	pytest

run:
	python code/flask_predict.py

docker-build:
	docker build -t gradeclass-service .

