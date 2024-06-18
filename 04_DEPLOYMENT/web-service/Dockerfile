FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# do stuff here
RUN pip install -U pip
RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["homework_04.py", "./"]

CMD ["python", "./homework_04.py"]