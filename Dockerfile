FROM python:3.11-slim

ENV APP_USER_NAME=appuser
ENV APP_USER_UID=1000
ENV APP_GROUP_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends gosu && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r "${APP_USER_NAME}" -g "${APP_GROUP_GID}" && \
    useradd -r -g "${APP_USER_NAME}" -u "${APP_USER_UID}" -s /bin/bash "${APP_USER_NAME}"

WORKDIR /app

COPY requirements/base.txt .
RUN pip install --no-cache-dir -r base.txt

COPY text_classifier/ ./text_classifier/

RUN mkdir -p /app/own_data /app/models && \
    chown -R "${APP_USER_NAME}":"${APP_USER_NAME}" /app/own_data /app/models

ENV PYTHONUNBUFFERED=1

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "text_classifier.agent"]