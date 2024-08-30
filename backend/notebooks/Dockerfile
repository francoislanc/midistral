FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

RUN git clone https://github.com/sshlien/abcmidi && cd abcmidi && make && make install
RUN apt update && apt install -y timidity
RUN useradd -ms /bin/bash dev
USER dev
WORKDIR /home/dev

ENV PATH="$PATH:/home/dev/.local/bin"

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/home/dev --ip 0.0.0.0 --no-browser"]
