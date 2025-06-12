# Dockerfile for Vector DB Challenge (FastAPI + uv + Cohere)
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set workdir
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Expose port (FastAPI default)
EXPOSE 8000
ENV PYTHONUNBUFFERED=1

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

CMD ["fastapi", "run", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]