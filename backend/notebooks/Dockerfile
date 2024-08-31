FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

RUN git clone https://github.com/sshlien/abcmidi && cd abcmidi && make && make install
RUN apt update && apt install -y timidity
RUN useradd -ms /bin/bash dev
USER dev
WORKDIR /home/dev

ENV PATH="$PATH:/home/dev/.local/bin"

RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install nvidia-cudnn-cu11

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/home/dev --ip 0.0.0.0 --no-browser"]
