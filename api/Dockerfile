FROM python:3.6
COPY ./bugan .
COPY ./.netrc /root/.netrc
RUN pip install --upgrade pip && \
    pip install --upgrade git+https://github.com/buganart/BUGAN.git#egg=bugan && \
    pip install flask
EXPOSE 8080
CMD ["python", "./api.py"]
