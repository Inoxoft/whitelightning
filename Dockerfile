FROM python:3.11-slim

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/* && \
    groupadd -r appuser -g ${GROUP_ID} && useradd -r -g appuser -u ${USER_ID} -s /bin/bash appuser

WORKDIR /app

COPY requirements/base.txt .
RUN pip install --no-cache-dir -r base.txt

COPY text_classifier/ ./text_classifier/

RUN mkdir -p /app/own_data /app/models && \
    chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []