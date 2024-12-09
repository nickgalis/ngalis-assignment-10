
APP_NAME = app.py

.PHONY: install run

run:
	python app.py

install:
	pip install -r requirements.txt