
FROM python:3.10

WORKDIR /app

COPY ./frontend/requirements.txt /app/requirements.txt

# --no-cache-dir stops cache from being stored in container
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy ./api contents to container workdir
COPY ./frontend /app

EXPOSE 8501


# run the Streamlit app
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]





