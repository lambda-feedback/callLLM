ARG BASE_VERSION=python:shimmy-socket-test-3.12
FROM ghcr.io/lambda-feedback/evaluation-function-base/$BASE_VERSION AS builder

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --without dev --no-root

FROM ghcr.io/lambda-feedback/evaluation-function-base/$BASE_VERSION

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy the evaluation function to the app directory
COPY evaluation_function ./evaluation_function

# Precompile python files for faster startup
RUN python -m compileall -q .

# Command to start the evaluation function with
ENV FUNCTION_COMMAND="python"

# Args to start the evaluation function with
ENV FUNCTION_ARGS="-m,evaluation_function.main"

# The transport to use for the RPC server
ENV FUNCTION_RPC_TRANSPORT="stdio"

ENV FUNCTION_WORKER_SEND_TIMEOUT="120s"

ENV LOG_LEVEL="debug"
