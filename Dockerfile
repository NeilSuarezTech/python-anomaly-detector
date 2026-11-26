FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "anomaly_detector.py"]
CMD ["sample_data/sensor_readings.csv", "--out", "output"]
