# ---------- build stage -----------------------------------------------------
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx  /bin/

ARG PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        pandoc unrtf gcc g++ make curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"
RUN pip install --upgrade pip

WORKDIR /src
COPY . .

RUN uv sync

RUN pip install ftfy
RUN pip install spacy
# Download spaCy Ukrainian models (small & transformer)
RUN python -m spacy download uk_core_news_sm \
 && python -m spacy download uk_core_news_trf

# ---------- runtime stage ---------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy site-packages and project code
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /src /app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=build /usr/bin/pandoc /usr/bin/unrtf /usr/bin/

ENTRYPOINT ["python", "-m", "processors.entities", "--use-pretrained", "process_batch"]
CMD ["--help"]
