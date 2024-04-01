# Ubuntu 20.04를 베이스 이미지로 사용
FROM ubuntu:20.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive

# 기본적인 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        curl \
        git \
        openjdk-11-jdk \
        python3.9 \
        python3-pip \
        python3.9-dev

# Python 3.9를 기본 Python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --set python /usr/bin/python3.9 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# TensorFlow GPU를 설치하기 위한 CUDA 및 cuDNN 설정 (예시는 CUDA 11.2 및 cuDNN 8에 대한 것임)
# TensorFlow의 호환 버전에 맞춰 CUDA와 cuDNN 버전을 조정해야 할 수도 있습니다.
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb && \
    apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub && \
    apt-get update && \
    apt-get -y install cuda

# Spark 설치
ARG SPARK_VERSION=3.0.0
ARG HADOOP_VERSION=2.7
RUN wget --no-verbose https://downloads.apache.org/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz && \
    tar -xzf spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz -C /opt && \
    mv /opt/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION /opt/spark

# 환경 변수에 Spark 추가
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# requirement.txt 파일을 도커 이미지로 복사
COPY requirement.txt /tmp/

# requirement.txt에 정의된 파이썬 라이브러리 설치
RUN pip install --no-cache-dir -r /tmp/requirement.txt

# 포트 설정 (필요에 따라 수정)
EXPOSE 8888 4040

# 시작 명령어 설정 (필요에 따라 변경)
CMD ["/bin/bash"]
