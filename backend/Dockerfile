FROM python:3.11.3

RUN git clone https://github.com/sshlien/abcmidi && cd abcmidi && make && make install
RUN apt update && apt install -y timidity

# Configure Poetry
ENV POETRY_VERSION=1.8.3
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

ENV APP_HOME /root
WORKDIR $APP_HOME
COPY /src $APP_HOME/src
COPY pyproject.toml $APP_HOME
COPY poetry.lock $APP_HOME
RUN poetry install --without audio-analysis 

EXPOSE 8000
CMD ["poetry", "run", "fastapi", "run", "src/midistral/serve.py", "--proxy-headers"]