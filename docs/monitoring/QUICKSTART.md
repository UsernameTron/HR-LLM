# Monitoring Quick Start Guide

## 1. Local Development Setup

### Start Monitoring Stack
```bash
# Start Prometheus
docker run -d \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Start Grafana
docker run -d \
  -p 3000:3000 \
  -v $(pwd)/grafana:/etc/grafana \
  grafana/grafana
```

### Access Dashboards
- Grafana: http://localhost:3000
- Default credentials: admin/admin

## 2. Key Metrics to Watch

### Performance
- P95 Latency: Should be <500ms
- Throughput: Monitor for sudden drops
- Error Rate: Should be <5%

### Resources
- Memory Usage: Warning at 33.6GB
- Cache Hit Rate: Should be >80%
- CPU Usage: Monitor for spikes

### Quality
- Attribution Entropy: 0.5-2.5 range
- Attribution Sparsity: 0.1-0.9 range

## 3. Common Actions

### View Metrics
1. Open Grafana dashboard
2. Select environment
3. Choose time range
4. Apply any filters

### Handle Alerts
1. Check alert details
2. Review related metrics
3. Follow runbook
4. Document resolution

### Update Thresholds
1. Edit dashboard JSON
2. Update alert rules
3. Test changes
4. Commit updates

## 4. Development Best Practices

### Adding New Metrics
```python
# In your code
from monitoring.config import metrics

# Counter example
request_counter = metrics.counter(
    'sentiment_requests_total',
    'Total number of sentiment requests'
)

# Histogram example
latency_histogram = metrics.histogram(
    'sentiment_request_duration_seconds',
    'Request duration in seconds',
    buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0]
)
```

### Testing Metrics
```python
# In your tests
async def test_metrics_collection():
    # Generate some load
    await generate_test_load()
    
    # Verify metrics
    metrics_data = await fetch_metrics()
    assert 'sentiment_requests_total' in metrics_data
```

## 5. Troubleshooting

### Metric Not Showing
1. Check metric name
2. Verify collection
3. Check Prometheus target
4. Review logs

### Dashboard Issues
1. Verify data source
2. Check query syntax
3. Review time range
4. Clear browser cache

### Alert Problems
1. Check threshold values
2. Verify alert conditions
3. Review alert history
4. Check notification channel

## 6. Getting Help
- Review main documentation
- Check system logs
- Contact DevOps team
- Open GitHub issue
