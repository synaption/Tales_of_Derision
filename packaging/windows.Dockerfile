FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12.7

RUN dpkg --add-architecture i386 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        wine64 \
        wine32 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

ENV WINEDEBUG=-all \
    WINEPREFIX=/opt/wine-python \
    WINEARCH=win64 \
    DISPLAY=:99

RUN Xvfb :99 -screen 0 1024x768x16 >/tmp/xvfb.log 2>&1 & \
    wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-amd64.exe" -O /tmp/python-installer.exe \
    && wine /tmp/python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 TargetDir=C:\\Python312 \
    && rm /tmp/python-installer.exe \
    && wine C:\\Python312\\python.exe -m pip --version
