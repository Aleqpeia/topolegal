# ---------- build stage -----------------------------------------------------
FROM python:3.12-slim AS build

ARG PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# system libs: pandoc, unrtf, build deps (gcc for spacy-transformers)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        pandoc unrtf gcc g++ make curl git \
    && rm -rf /var/lib/apt/lists/*

# optional: faster wheels for spacy-transformers / torch
RUN pip install --upgrade pip

# Copy project
WORKDIR /src
COPY . .

# Install Python deps (PEP-517 build)
RUN pip install .[all]

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

# copy pandoc + unrtf binaries from build stage
COPY --from=build /usr/bin/pandoc /usr/bin/unrtf /usr/bin/

# default command (override at run-time)
ENTRYPOINT ["python", "-m", "processors.entities"]
CMD ["--help"]
