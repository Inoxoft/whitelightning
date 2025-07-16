FROM python:3.11-slim


WORKDIR /app


COPY requirements/base.txt .


RUN pip install --no-cache-dir -r base.txt


COPY text_classifier/ ./text_classifier/


RUN mkdir -p /app/own_data


ENV PYTHONUNBUFFERED=1


ENTRYPOINT ["python", "-m", "text_classifier.agent"]

CMD []