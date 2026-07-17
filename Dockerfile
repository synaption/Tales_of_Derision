FROM python:3.12-slim

ARG GAME_COMMIT=unknown
ARG TTYD_VERSION=1.7.7

LABEL org.opencontainers.image.revision="${GAME_COMMIT}"

ENV PYTHONUNBUFFERED=1
ENV TERM=xterm-256color
ENV GAME_COMMIT="${GAME_COMMIT}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin game

RUN arch="$(dpkg --print-architecture)" \
    && case "$arch" in \
      amd64) ttyd_arch="x86_64" ;; \
      arm64) ttyd_arch="aarch64" ;; \
      *) echo "Unsupported architecture: $arch" && exit 1 ;; \
    esac \
    && curl -fsSL \
      "https://github.com/tsl0922/ttyd/releases/download/${TTYD_VERSION}/ttyd.${ttyd_arch}" \
      -o /usr/local/bin/ttyd \
    && chmod 0755 /usr/local/bin/ttyd

WORKDIR /opt/game

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN chown -R game:game /opt/game

USER game

EXPOSE 7681

ENTRYPOINT ["ttyd", "--writable", "--port", "7681", "python3", "src/main.py"]
