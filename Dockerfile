# syntax=docker/dockerfile:1

FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime 

COPY torch-env.yml .

RUN conda env create -f torch-env.yml

RUN echo "conda activate torch-env" > ~/.bashrc

ENTRYPOINT ["/bin/bash"]
