FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY monitoring/metrics_exporter.py monitoring/
COPY monitoring/grafana/dashboards monitoring/grafana/dashboards

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=9091

# Expose metrics port
EXPOSE 9091

# Run metrics exporter
CMD ["python", "monitoring/metrics_exporter.py", "--port", "9091"]
