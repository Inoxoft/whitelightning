FROM python:3.11-slim

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -r appuser -g ${GROUP_ID} && useradd -r -g appuser -u ${USER_ID} appuser

WORKDIR /app

COPY requirements/base.txt .

RUN pip install --no-cache-dir -r base.txt

COPY text_classifier/ ./text_classifier/

RUN mkdir -p /app/own_data /app/models && \
    chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1

RUN echo '#!/bin/bash\n\
if [ -d "/app/models" ]; then\n\
    chmod 777 /app/models 2>/dev/null || true\n\
    find /app/models -type d -exec chmod 777 {} \; 2>/dev/null || true\n\
fi\n\
exec python -m text_classifier.agent "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD []