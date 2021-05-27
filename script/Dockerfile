FROM python:3.8.5
COPY ./predict.py ./predict.py
COPY ./.netrc /root/.netrc
RUN pip install --upgrade pip && \
    pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --upgrade git+https://github.com/buganart/BUGAN.git#egg=bugan
ENTRYPOINT ["python", "./predict.py"]
