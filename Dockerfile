# Example taken from https://github.com/astral-sh/uv-docker-example/blob/main/multistage.Dockerfile

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Bytecode compilation, copy from the cache instead of linking, since it is
# a mounted volume
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
# across both images. 
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-editable --all-extras

ADD python_repo_template /app/python_repo_template
ADD main.py /app/main.py
ADD pyproject.toml /app/pyproject.toml

# Copy the lock file to make sure the Docker environment has the same 
# dependencies as the development environment
ADD uv.lock /app/uv.lock 

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --all-extras


# Then, use a final image without uv
# It is important to use the image that matches the builder, as the path to the
# Python executable must be the same
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy the application from the builder
COPY --from=builder --chown=app:app /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "main.py", "--number", "10"]
