FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12.7

RUN dpkg --add-architecture i386 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        wine \
        wine64 \
        wine32 \
        xauth \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

ENV WINEDEBUG=-all \
    WINEDLLOVERRIDES="winemenubuilder.exe=d;winedbg.exe=d" \
    WINEPREFIX=/opt/wine-python \
    WINEARCH=win64 \
    DISPLAY=:99

RUN wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-amd64.exe" -O /tmp/python-installer.exe \
    && xvfb-run -a wine /tmp/python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 TargetDir=C:\\Python312 \
    && rm /tmp/python-installer.exe \
    && xvfb-run -a wine C:\\Python312\\python.exe -m pip --version
