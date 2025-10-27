FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates

RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY . /app
WORKDIR /app

RUN uv venv -q

RUN uv sync

CMD ["uv", "run", "./scripts/eval.py", $name]
