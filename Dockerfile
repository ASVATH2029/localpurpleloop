# The hard sandbox guarantee (plan section 3).
#
# Build:
#   docker build -t purple-loop .
#
# Run with NO network at all, so even a runaway loop or a badly biased
# mutation policy cannot reach anything outside this container:
#   docker run --rm --network none -v "$PWD/out:/app/out" purple-loop
#
# The loop needs no credentials and makes no outbound calls, so --network none
# costs nothing and turns "it is sandboxed" from a claim into a property.
FROM python:3.12-slim

WORKDIR /app

# deps first, so edits to the source do not invalidate the pip layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml pytest.ini ./
COPY purpleloop ./purpleloop
COPY tests ./tests

# fail the build if the leakage guards are not holding
RUN python -m pytest -q

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "purpleloop", "run"]
