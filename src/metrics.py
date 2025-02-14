"""
Centralized metrics management for the hiring sentiment tracker.
"""
import logging
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, REGISTRY

logger = logging.getLogger(__name__)

class MetricsManager:
    """
    Centralized metrics management to prevent duplication and ensure proper initialization
    """
    def __init__(self, registry=REGISTRY):
        self.registry = registry
        # Internal state tracking
        self._hits = 0
        self._misses = 0
        self._initialize_metrics()
        
        # Initialize effectiveness to 0
        self.cache_effectiveness.set(0)
    
    def _initialize_metrics(self):
        """Initialize all metrics with proper labeling and documentation"""
        try:
            # Business Intelligence Metrics
            self.sentiment_by_department = Gauge(
                'hiring_sentiment_by_department',
                'Average sentiment score by department',
                ['department', 'timeframe'],
                registry=self.registry
            )
            # Initialize with zero for common departments
            for dept in ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']:
                self.sentiment_by_department.labels(department=dept, timeframe='current').set(0)
            
            self.skill_demand_trend = Gauge(
                'hiring_skill_demand_trend',
                'Trend in skill demand over time',
                ['skill', 'timeframe'],
                registry=self.registry
            )
            # Initialize with zero for common skills
            for skill in ['Python', 'Java', 'JavaScript', 'AWS', 'Kubernetes']:
                self.skill_demand_trend.labels(skill=skill, timeframe='current').set(0)
            
            self.market_competitiveness = Gauge(
                'hiring_market_competitiveness',
                'Market competitiveness score',
                ['department', 'region'],
                registry=self.registry
            )
            # Initialize with zero for common departments
            for dept in ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']:
                self.market_competitiveness.labels(department=dept, region='global').set(0)
            
            self.prediction_accuracy = Gauge(
                'hiring_prediction_accuracy',
                'Model prediction accuracy',
                ['prediction_type'],
                registry=self.registry
            )
            # Initialize with zero for sentiment prediction
            self.prediction_accuracy.labels(prediction_type='sentiment').set(0)
            
            # Hiring Wave Indicator
            self.hiring_wave = Gauge(
                'hiring_wave_indicator',
                'Indicator of hiring waves',
                ['department', 'timeframe'],
                registry=self.registry
            )
            # Initialize with zero
            for dept in ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']:
                self.hiring_wave.labels(department=dept, timeframe='current').set(0)
            
            # Technical Performance Metrics
            self.response_time = Histogram(
                'hiring_response_time_seconds',
                'Response time distribution',
                buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0],
                registry=self.registry
            )
            # Cache hit tracking
            self.cache_hits = Counter(
                'hiring_cache_hits_total',
                'Total number of cache hits',
                registry=self.registry
            )
            self.cache_misses = Counter(
                'hiring_cache_misses_total',
                'Total number of cache misses',
                registry=self.registry
            )
            self.cache_effectiveness = Gauge(
                'hiring_cache_effectiveness_ratio',
                'Cache hit ratio percentage',
                registry=self.registry
            )
            # Initialize effectiveness to 0
            self.cache_effectiveness.set(0)
            
            # Initialize counters to 0 to ensure they exist
            self.cache_hits.inc(0)
            self.cache_misses.inc(0)
            
            self.error_rate = Gauge(
                'hiring_error_rate_percentage',
                'Error rate percentage',
                ['error_type'],
                registry=self.registry
            )
            # Initialize with zero for common error types
            for error_type in ['analysis', 'cache', 'processing']:
                self.error_rate.labels(error_type=error_type).set(0)
            
            # Model Performance Metrics
            self.model_inference_time = Histogram(
                'hiring_model_inference_seconds',
                'Model inference time',
                buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25],
                registry=self.registry
            )
            
            # Confidence Distribution
            self.confidence_distribution = Histogram(
                'hiring_confidence_distribution',
                'Distribution of model confidence scores',
                ['analysis_type'],
                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                registry=self.registry
            )
            # Initialize a sample for sentiment analysis type
            self.confidence_distribution.labels(analysis_type='sentiment').observe(0.0)
            
            # Skill Premium
            self.skill_premium = Gauge(
                'hiring_skill_premium',
                'Premium score for various skills',
                ['skill', 'timeframe'],
                registry=self.registry
            )
            # Initialize with zero for common skills
            for skill in ['Python', 'Java', 'JavaScript', 'AWS', 'Kubernetes']:
                self.skill_premium.labels(skill=skill, timeframe='current').set(0)
            
        except ValueError as e:
            # Log but don't fail - this helps with test environments
            logger.warning(f"Metric already registered: {str(e)}")
    
    def get_metric_value(self, metric_name: str) -> float:
        """
        Retrieve the metric value for the sample with no labels.
        """
        metrics = {m.name: m for m in self.registry.collect()}
        metric = metrics.get(metric_name)
        if metric:
            # Find the sample with an empty labels dict
            for sample in metric.samples:
                if sample.labels == {}:
                    return sample.value
        return 0.0
        
    def get_cache_metrics_snapshot(self) -> dict:
        """
        Get a complete snapshot of current cache metrics.
        
        Returns:
            dict: Dictionary containing current cache metrics including:
                - hits: Total number of cache hits
                - misses: Total number of cache misses
                - total_requests: Total number of cache requests
                - effectiveness: Cache hit ratio as a percentage
                - internal_hits: Internal counter for hits
                - internal_misses: Internal counter for misses
        """
        hits = self.get_metric_value('hiring_cache_hits_total')
        misses = self.get_metric_value('hiring_cache_misses_total')
        effectiveness = self.get_metric_value('hiring_cache_effectiveness_ratio')
        
        snapshot = {
            'hits': hits,
            'misses': misses,
            'total_requests': hits + misses,
            'effectiveness': effectiveness,
            'internal_hits': self._hits,
            'internal_misses': self._misses
        }
        
        logger.debug(f"Cache metrics snapshot: {snapshot}")
        return snapshot

    def update_cache_metrics(self, hit: bool | None = None):
        """
        Update cache hit/miss metrics and effectiveness ratio.
        
        Args:
            hit: Optional[bool] - If True, increments cache hits. If False, increments cache misses.
                                If None, only updates the effectiveness ratio.
        """
        if hit is not None:
            if hit:
                self.cache_hits.inc()
                self._hits += 1
                logger.debug(f"Cache hit recorded. Total hits: {self._hits}")
            else:
                self.cache_misses.inc()
                self._misses += 1
                logger.debug(f"Cache miss recorded. Total misses: {self._misses}")
            
            # Always calculate and update effectiveness after any hit/miss
            total = self._hits + self._misses
            if total > 0:
                effectiveness = (self._hits / total) * 100.0
                # Ensure we're setting a non-zero value when we have hits
                if self._hits > 0:
                    self.cache_effectiveness.set(effectiveness)
                    logger.debug(
                        f"Cache metrics updated - Hits: {self._hits}, "
                        f"Misses: {self._misses}, "
                        f"Total: {total}, "
                        f"Effectiveness: {effectiveness:.2f}%"
                    )
