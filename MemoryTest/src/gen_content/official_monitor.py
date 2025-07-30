#### Memory Performance Monitoring
# from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events import (
    MemoryQueryCompletedEvent,
    MemorySaveCompletedEvent
)
import time

class MemoryPerformanceMonitor(BaseEventListener):
    def __init__(self):
        super().__init__()
        self.query_times = []
        self.save_times = []

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(MemoryQueryCompletedEvent)
        def on_memory_query_completed(source, event: MemoryQueryCompletedEvent):
            self.query_times.append(event.query_time_ms)
            print(f"Memory query completed in {event.query_time_ms:.2f}ms. Query: '{event.query}'")
            print(f"Average query time: {sum(self.query_times)/len(self.query_times):.2f}ms")

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_memory_save_completed(source, event: MemorySaveCompletedEvent):
            self.save_times.append(event.save_time_ms)
            print(f"Memory save completed in {event.save_time_ms:.2f}ms")
            print(f"Average save time: {sum(self.save_times)/len(self.save_times):.2f}ms")

# Create an instance of your listener
# memory_monitor = MemoryPerformanceMonitor()

####### Memory Performance Logger #######
from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events import (
    MemorySaveStartedEvent,
    MemoryQueryStartedEvent,
    MemoryRetrievalCompletedEvent
)
import logging

# Configure logging
logger = logging.getLogger('memory_events')

class MemoryLogger(BaseEventListener):
    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_memory_save_started(source, event: MemorySaveStartedEvent):
            if event.agent_role:
                logger.info(f"Agent '{event.agent_role}' saving memory: {event.value[:50]}...")
            else:
                logger.info(f"Saving memory: {event.value[:50]}...")

        @crewai_event_bus.on(MemoryQueryStartedEvent)
        def on_memory_query_started(source, event: MemoryQueryStartedEvent):
            logger.info(f"Memory query started: '{event.query}' (limit: {event.limit})")

        @crewai_event_bus.on(MemoryRetrievalCompletedEvent)
        def on_memory_retrieval_completed(source, event: MemoryRetrievalCompletedEvent):
            if event.task_id:
                logger.info(f"Memory retrieved for task {event.task_id} in {event.retrieval_time_ms:.2f}ms")
            else:
                logger.info(f"Memory retrieved in {event.retrieval_time_ms:.2f}ms")
            logger.debug(f"Memory content: {event.memory_content}")

# Create an instance of your listener
# memory_logger = MemoryLogger()

from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events import (
    MemorySaveFailedEvent,
    MemoryQueryFailedEvent
)
import logging
from typing import Optional

# # Configure logging
logger = logging.getLogger('memory_errors')

class MemoryErrorTracker(BaseEventListener):
    def __init__(self, notify_email: Optional[str] = None):
        super().__init__()
        self.notify_email = notify_email
        self.error_count = 0

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(MemorySaveFailedEvent)
        def on_memory_save_failed(source, event: MemorySaveFailedEvent):
            self.error_count += 1
            agent_info = f"Agent '{event.agent_role}'" if event.agent_role else "Unknown agent"
            error_message = f"Memory save failed: {event.error}. {agent_info}"
            logger.error(error_message)

            if self.notify_email and self.error_count % 5 == 0:
                self._send_notification(error_message)

        @crewai_event_bus.on(MemoryQueryFailedEvent)
        def on_memory_query_failed(source, event: MemoryQueryFailedEvent):
            self.error_count += 1
            error_message = f"Memory query failed: {event.error}. Query: '{event.query}'"
            logger.error(error_message)

            if self.notify_email and self.error_count % 5 == 0:
                self._send_notification(error_message)

    def _send_notification(self, message):
        # Implement your notification system (email, Slack, etc.)
        print(f"[NOTIFICATION] Would send to {self.notify_email}: {message}")

# # Create an instance of your listener
# error_tracker = MemoryErrorTracker(notify_email="admin@example.com")

## Analyze 
from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events import (
    MemoryQueryCompletedEvent,
    MemorySaveCompletedEvent
)

class MemoryAnalyticsForwarder(BaseEventListener):
    def __init__(self, analytics_client):
        super().__init__()
        self.client = analytics_client

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(MemoryQueryCompletedEvent)
        def on_memory_query_completed(source, event: MemoryQueryCompletedEvent):
            # Forward query metrics to analytics platform
            self.client.track_metric({
                "event_type": "memory_query",
                "query": event.query,
                "duration_ms": event.query_time_ms,
                "result_count": len(event.results) if hasattr(event.results, "__len__") else 0,
                "timestamp": event.timestamp
            })

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_memory_save_completed(source, event: MemorySaveCompletedEvent):
            # Forward save metrics to analytics platform
            self.client.track_metric({
                "event_type": "memory_save",
                "agent_role": event.agent_role,
                "duration_ms": event.save_time_ms,
                "timestamp": event.timestamp
            })

# usage
