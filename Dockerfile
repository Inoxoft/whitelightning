FROM python:3.11-slim


RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

WORKDIR /app


COPY requirements/base.txt .


RUN pip install --no-cache-dir -r base.txt


COPY text_classifier/ ./text_classifier/


RUN mkdir -p /app/own_data /app/models && \
    chown -R appuser:appuser /app


USER appuser

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "text_classifier.agent"]
CMD []