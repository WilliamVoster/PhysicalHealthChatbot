
FROM python:3.10

WORKDIR /app

COPY ./api/requirements.txt /app/requirements.txt

# --no-cache-dir stops cache from being stored in container
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy ./api contents to container workdir
COPY ./api /app

EXPOSE 8000


# run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]





