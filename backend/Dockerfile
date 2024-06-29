FROM python:3.11.3
ENV PYTHONUNBUFFERED True

RUN git clone https://github.com/sshlien/abcmidi && cd abcmidi && make && make install
RUN apt update && apt install -y timidity

RUN pip install --upgrade pip
RUN pip install poetry

ENV APP_HOME /root
WORKDIR $APP_HOME
COPY /src $APP_HOME/src
COPY pyproject.toml $APP_HOME
COPY poetry.lock $APP_HOME
RUN poetry install

EXPOSE 8000
CMD ["poetry", "run", "fastapi", "run", "src/midistral/serve.py", "--proxy-headers"]