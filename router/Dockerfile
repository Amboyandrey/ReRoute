FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev

ENV ROUTER_PORT=9000
EXPOSE 9000
CMD ["uv", "run", "reroute-serve"]
