FROM python:3.10-slim

WORKDIR /backend

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY backend/requirements_backend.txt .
RUN pip install -r requirements_backend.txt

COPY backend/ .

COPY Data /Data
COPY Models /Models
COPY Results /Results

EXPOSE 8000

CMD ["sh", "-c", "python seed_db.py && uvicorn api.main:app --host 0.0.0.0 --port 8000"]