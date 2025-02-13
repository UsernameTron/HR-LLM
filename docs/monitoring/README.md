# Sentiment Analysis Monitoring System

## Overview
This document outlines the monitoring infrastructure for the Sentiment Analysis Pipeline, including metrics collection, visualization, and alerting.

## Key Components

### 1. Metrics Collection
- **Location**: `src/monitoring/`
- **Key Files**:
  - `config.py`: Metrics configuration and registry setup
  - `metrics_server.py`: Prometheus metrics endpoint

#### Core Metrics
- Request Latency (P95)
- Message Throughput
- Memory Usage
- Cache Hit Rates
- Attribution Quality
- Error Rates by Type

### 2. Grafana Dashboards
- **Location**: `grafana/dashboards/`
- **Main Dashboard**: `sentiment-analysis.json`

#### Dashboard Sections
1. **Performance Metrics (Top Row)**
   - P95 Latency Gauge (Warning: 500ms)
   - Throughput Graph
   - Error Rate Gauge (Warning: 5%)

2. **Resource Utilization (Middle Row)**
   - Memory Usage (Warning: 33.6GB, Critical: 40.8GB)
   - Cache Performance
   - System Health Status

3. **Quality Metrics (Bottom Row)**
   - Attribution Quality Trends
   - Error Type Distribution
   - System Health Grid

### 3. Alert Configuration
- **Location**: `grafana/alerts/rules.yml`

#### Alert Rules
1. **Memory Usage**
   - Warning: >33.6GB
   - Critical: >40.8GB
   - Duration: 5m

2. **Error Rates**
   - Warning: >5%
   - Duration: 2m

3. **Attribution Quality**
   - Entropy Range: 0.5-2.5
   - Sparsity Range: 0.1-0.9
   - Duration: 5m

4. **Performance**
   - P95 Latency: >1s
   - Cache Hit Rate: <50%

## Setup Instructions

### Prerequisites
- Prometheus Server
- Grafana Server (v8.0+)
- Access to metrics endpoint

### Installation Steps
1. **Prometheus Setup**
   ```yaml
   scrape_configs:
     - job_name: 'sentiment-analysis'
       metrics_path: '/metrics'
       static_configs:
         - targets: ['localhost:8000']
   ```

2. **Grafana Configuration**
   ```bash
   # Copy dashboard provisioning
   cp grafana/provisioning/dashboards/* /etc/grafana/provisioning/dashboards/
   
   # Copy dashboard definition
   cp grafana/dashboards/sentiment-analysis.json /etc/grafana/dashboards/
   ```

3. **Alert Setup**
   ```bash
   # Copy alert rules
   cp grafana/alerts/rules.yml /etc/prometheus/rules/
   ```

## Maintenance

### Daily Checks
1. Monitor memory usage trends
2. Review error rate patterns
3. Check attribution quality metrics
4. Verify cache performance

### Weekly Tasks
1. Review alert history
2. Analyze performance trends
3. Update thresholds if needed
4. Backup dashboard configurations

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check batch processing size
   - Verify memory leaks
   - Review cache size

2. **High Error Rates**
   - Check log files
   - Review error types
   - Verify input data quality

3. **Poor Attribution Quality**
   - Review model inputs
   - Check preprocessing steps
   - Verify model weights

## Best Practices

1. **Metric Collection**
   - Use consistent naming
   - Include relevant labels
   - Set appropriate intervals

2. **Dashboard Management**
   - Regular backups
   - Version control
   - Document changes

3. **Alert Configuration**
   - Avoid alert fatigue
   - Set meaningful thresholds
   - Include clear descriptions

## Support
For issues or questions:
1. Check troubleshooting guide
2. Review system logs
3. Contact system administrators
