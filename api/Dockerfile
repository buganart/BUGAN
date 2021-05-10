FROM python:3.6
COPY ./bugan .
COPY ./.netrc /root/.netrc
RUN pip install --upgrade pip && \
    pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --upgrade git+https://github.com/buganart/BUGAN.git#egg=bugan && \
    pip install flask
EXPOSE 8080
ENTRYPOINT ["python", "./api.py"]
