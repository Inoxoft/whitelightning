FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

COPY requirements/base.txt /app/requirements/base.txt
RUN pip install --no-cache-dir -r /app/requirements/base.txt

COPY text_classifier/ /app/text_classifier/

ENV PYTHONPATH="/app:${PYTHONPATH}"

RUN python --version
RUN pip list

CMD ["bash"]
