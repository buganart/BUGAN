FROM python:3.6
COPY ./checkpoint ./checkpoint
COPY ./predict.py ./predict.py
RUN pip install --upgrade pip && \
    pip install torch==1.8.1 && \
    pip install --upgrade git+https://github.com/buganart/BUGAN.git#egg=bugan
ENTRYPOINT ["python", "./predict.py"]
