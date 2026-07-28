FROM python:3.10-slim
WORKDIR /code
COPY backend/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend /code
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
