FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN echo 'deb http://deb.debian.org/debian bookworm-backports main' \
        > /etc/apt/sources.list.d/backports.list \
    && apt-get update \
    && apt-get install -y curl \
    && apt-get install -y -t bookworm-backports openjdk-21-jdk-headless \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV UV_LINK_MODE=copy

COPY pyproject.toml /workspace/
COPY src/ /workspace/src/

RUN uv venv /opt/venv --python python3.12 \
    && . /opt/venv/bin/activate \
    && uv pip install .

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

CMD ["/bin/bash"]
