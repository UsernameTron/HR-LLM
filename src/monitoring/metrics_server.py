"""Prometheus metrics server for the sentiment analysis pipeline."""
import asyncio
import logging
import signal
from aiohttp import web
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from .config import metrics, registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def metrics_handler(request):
    """Expose Prometheus metrics."""
    try:
        metrics_data = generate_latest(registry)
        resp = web.Response(body=metrics_data)
        resp.headers['Content-Type'] = CONTENT_TYPE_LATEST
        return resp
    except Exception as e:
        logger.error(f"Error generating metrics: {str(e)}")
        raise web.HTTPInternalServerError(text="Error generating metrics")

class MetricsServer:
    """Metrics server with graceful shutdown support."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, host='0.0.0.0', port=9090):
        if self._initialized:
            return

        self.host = host
        self.port = port
        self.app = web.Application()
        self.app.router.add_get('/metrics', metrics_handler)
        self.app.router.add_get('/health', self.health_check)
        self.runner = None
        self.site = None
        self._initialized = True
        
        # Setup signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)
        
    async def start(self):
        """Start the metrics server."""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            logger.info(f"Metrics server running at http://{self.host}:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")
            await self.cleanup()
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}. Starting graceful shutdown...")
        asyncio.create_task(self.cleanup())

    async def health_check(self, request):
        """Health check endpoint."""
        return web.Response(text="healthy")

    async def cleanup(self):
        """Cleanup server resources."""
        try:
            if self.site:
                await self.site.stop()
                logger.info("Stopped metrics server site")
            
            if self.runner:
                await self.runner.cleanup()
                logger.info("Cleaned up server runner")
                
            # Reset instance state
            self.site = None
            self.runner = None
            self._initialized = False
            
            logger.info("Server resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

async def start_metrics_server(host='0.0.0.0', port=9090):
    """Start the metrics server."""
    server = MetricsServer(host, port)
    await server.start()
    return server

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    async def run_server():
        server = await start_metrics_server()
        try:
            # Keep the server running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await server.cleanup()
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nShutting down metrics server...")