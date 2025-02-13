# ML System Architecture Diagrams

## 1. High-Level System Architecture

```mermaid
graph TB
    Client[Client Requests] --> LB[Load Balancer]
    LB --> API[API Gateway]
    
    subgraph Processing Pipeline
        API --> Queue[Redis Queue]
        Queue --> Workers[Worker Pool]
        Workers --> Model[ML Model]
        Model --> Cache[Redis Cache]
        Model --> MPS[Metal Performance Shaders]
    end
    
    subgraph Monitoring System
        Workers --> Metrics[Metrics Collector]
        Model --> Metrics
        Cache --> Metrics
        Metrics --> Prometheus[Prometheus]
        Prometheus --> Grafana[Grafana Dashboards]
        Grafana --> Alerts[Alert Manager]
    end
```

## 2. Data Flow Architecture

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API
    participant Q as Queue
    participant W as Worker
    participant M as Model
    participant Ca as Cache
    participant Me as Metrics

    C->>A: Submit Request
    A->>Q: Enqueue Message
    Q->>W: Dequeue Batch
    W->>Ca: Check Cache
    alt Cache Hit
        Ca-->>W: Return Cached Result
    else Cache Miss
        W->>M: Process Batch
        M-->>W: Return Results
        W->>Ca: Update Cache
    end
    W->>Me: Record Metrics
    W-->>C: Return Response
```

## 3. Monitoring Architecture

```mermaid
graph LR
    subgraph Application Components
        Model[ML Model] --> Collector
        Cache[Redis Cache] --> Collector
        Workers[Workers] --> Collector
    end
    
    subgraph Metrics Pipeline
        Collector[Metrics Collector]
        Collector --> Prometheus
        Prometheus[Prometheus] --> Grafana[Grafana]
        Prometheus --> Alerts[Alert Manager]
        Alerts --> Notify[Notifications]
    end
    
    subgraph Visualization
        Grafana --> P[Performance Dashboard]
        Grafana --> R[Resource Dashboard]
        Grafana --> Q[Quality Dashboard]
    end
```

## 4. Memory Management

```mermaid
graph TB
    subgraph System Memory
        RAM[48GB RAM]
        subgraph Active Memory
            App[Application: 648MB]
            Redis[Redis Cache]
            MPS[Metal Performance Shaders]
        end
    end
    
    subgraph Thresholds
        Warning[Warning: 33.6GB]
        Critical[Critical: 40.8GB]
    end
    
    App --> Monitor[Memory Monitor]
    Redis --> Monitor
    MPS --> Monitor
    Monitor --> Alert[Alert Manager]
```

## 5. Load Testing Pipeline

```mermaid
graph LR
    subgraph Test Configuration
        Config[Test Config]
        Params[Parameters]
    end
    
    subgraph Load Generation
        Generator[Load Generator]
        Queue[Message Queue]
    end
    
    subgraph Processing
        Workers[Worker Pool]
        Model[ML Model]
        Cache[Cache]
    end
    
    subgraph Metrics Collection
        Latency[Latency]
        Throughput[Throughput]
        Memory[Memory Usage]
        Quality[Attribution Quality]
    end
    
    Config --> Generator
    Params --> Generator
    Generator --> Queue
    Queue --> Workers
    Workers --> Model
    Model --> Cache
    Workers --> Latency
    Workers --> Throughput
    Model --> Memory
    Model --> Quality
```

## 6. Alert Flow

```mermaid
graph TB
    subgraph Metrics Sources
        M1[Memory Usage]
        M2[Error Rates]
        M3[Latency]
        M4[Quality Metrics]
    end
    
    subgraph Alert Processing
        Prometheus --> Rules[Alert Rules]
        Rules --> Manager[Alert Manager]
    end
    
    subgraph Notification
        Manager --> Slack[Slack]
        Manager --> Email[Email]
        Manager --> PagerDuty[PagerDuty]
    end
    
    M1 & M2 & M3 & M4 --> Prometheus
```

## Using These Diagrams

These architecture diagrams provide different views of our ML system:

1. **High-Level System Architecture**: Shows the main components and their interactions
2. **Data Flow Architecture**: Illustrates the request processing sequence
3. **Monitoring Architecture**: Details our observability setup
4. **Memory Management**: Visualizes our memory allocation and monitoring
5. **Load Testing Pipeline**: Shows how we generate and measure load
6. **Alert Flow**: Demonstrates how alerts are processed and delivered

### Key Points

1. **Scalability**
   - Worker pool can scale horizontally
   - Redis cache provides fast access to frequent requests
   - Metal Performance Shaders optimize ML operations

2. **Reliability**
   - Multiple monitoring layers
   - Comprehensive alert system
   - Cache fallback for high availability

3. **Performance**
   - Efficient memory usage (648MB total)
   - Low latency (55ms P95)
   - High throughput (500 msg/s)

4. **Monitoring**
   - Real-time metrics collection
   - Multi-level alerting
   - Comprehensive dashboards

### Implementation Notes

When implementing features or debugging issues, refer to these diagrams to understand:
- Component relationships
- Data flow paths
- Monitoring points
- Alert triggers

For specific metrics and thresholds, refer to the main documentation.
