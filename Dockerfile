# Same Python version that trains the model in CI
FROM python:3.13-slim

WORKDIR /app

# Install libraries first (this layer is cached, so rebuilds are fast)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API needs: the code and the approved model
COPY app.py champion.pkl ./

EXPOSE 8000

# 0.0.0.0 = accept requests from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
