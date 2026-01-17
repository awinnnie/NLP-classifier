FROM python:3.10-slim

WORKDIR /frontend

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY frontend/requirements_frontend.txt .
RUN pip install -r requirements_frontend.txt

COPY frontend/ .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
