# ML Performance Testing Quick Reference

## Common Operations

### 1. Start Load Test
```bash
# Run load test with default configuration
python tests/load/run_load_tests.py

# Run with custom parameters
python tests/load/run_load_tests.py --rate 500 --batch-size 128
```

### 2. Monitor Performance
```bash
# View Grafana dashboard
open http://localhost:3000/d/sentiment-analysis

# Check metrics endpoint
curl http://localhost:8000/metrics
```

### 3. Check System Health
```bash
# View system metrics
top -l 1 | grep "PhysMem"

# Check Redis memory
redis-cli info memory
```

### 4. Common Metrics

#### Performance Metrics
- P95 Latency: 55ms (target: <500ms)
- Throughput: 500 msg/s
- Memory: 648MB (1.35% of 48GB)
- Error Rate: 0%

#### Quality Metrics
- Attribution Sparsity: 1.00
- Attribution Entropy: 1.88
- Cache Hit Rate: >95%

### 5. Alert Thresholds

#### Warning Levels
- Memory: >33.6GB
- Error Rate: >5%
- Latency: >500ms
- Cache Miss: >20%

#### Critical Levels
- Memory: >40.8GB
- Error Rate: >10%
- Latency: >1000ms
- Cache Miss: >50%

### 6. Quick Troubleshooting

#### High Memory Usage
```bash
# Check memory allocation
ps aux | grep python | sort -nrk 4,4

# Clear Redis cache
redis-cli FLUSHALL
```

#### High Latency
```bash
# Check system load
uptime

# Monitor real-time latency
tail -f logs/performance.log | grep latency
```

#### Error Spikes
```bash
# View error logs
tail -f logs/error.log

# Check error metrics
curl http://localhost:8000/metrics | grep error
```

### 7. Useful Commands

#### Metrics Collection
```bash
# Export metrics snapshot
curl -o metrics_snapshot.txt http://localhost:8000/metrics

# Analyze metrics
python tools/analyze_metrics.py metrics_snapshot.txt
```

#### Performance Analysis
```bash
# Run quick health check
./tools/health_check.sh

# Generate performance report
python tools/generate_report.py
```

### 8. Key Files

#### Configuration
- `config/performance.yml`: Performance settings
- `config/monitoring.yml`: Monitoring configuration
- `grafana/dashboards/`: Dashboard definitions

#### Logs
- `logs/performance.log`: Performance metrics
- `logs/error.log`: Error tracking
- `logs/attribution.log`: Attribution quality

### 9. Contact

For urgent issues:
1. Check runbooks in `docs/runbooks/`
2. Review error patterns
3. Contact on-call engineer
