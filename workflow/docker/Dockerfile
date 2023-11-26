FROM continuumio/miniconda3@sha256:a89cd00402729eb4af1efb8424aec26f8ffe04d8f3250b1485f1814bd640d357

WORKDIR /code

# Setup our environment
COPY environment.yml environment.yml

ENV CONDA_EXE /opt/conda/condabin/conda
ENV CONDA_PREFIX /opt/conda/envs/embedgem
ENV CONDA_PYTHON_EXE /opt/conda/bin/python
ENV PATH /opt/conda/envs/embedgem/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV CONDA_DEFAULT_ENV embedgem
ENV CONDA_PROMPT_MODIFIER (embedgem)

RUN conda env create -n embedgem --file environment.yml --solver libmamba
