import logging
import os
from crewai.utilities.events.base_event_listener import BaseEventListener
from crewai.utilities.events import (
    MemoryQueryCompletedEvent,
    MemorySaveCompletedEvent,
    MemorySaveStartedEvent,
    MemoryQueryStartedEvent,
    MemoryRetrievalCompletedEvent,
    MemorySaveFailedEvent,
    MemoryQueryFailedEvent
)
from typing import Optional

# ✅ 设置日志文件路径
log_dir = "src/gen_content/log"
os.makedirs(log_dir, exist_ok=True)

# ✅ 配置 memory 性能日志
performance_log_path = os.path.join(log_dir, "memory_performance.log")
performance_logger = logging.getLogger("memory_performance")
performance_handler = logging.FileHandler(performance_log_path, encoding="utf-8")
performance_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
performance_logger.setLevel(logging.INFO)
performance_logger.addHandler(performance_handler)

# ✅ 配置 memory 事件日志
event_log_path = os.path.join(log_dir, "memory_events.log")
event_logger = logging.getLogger("memory_events")
event_handler = logging.FileHandler(event_log_path, encoding="utf-8")
event_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
event_logger.setLevel(logging.INFO)
event_logger.addHandler(event_handler)

# ✅ 配置 memory 错误日志
error_log_path = os.path.join(log_dir, "memory_errors.log")
error_logger = logging.getLogger("memory_errors")
error_handler = logging.FileHandler(error_log_path, encoding="utf-8")
error_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
error_logger.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)


# ✅ Memory Performance Monitor 写入日志
class MemoryPerformanceMonitor(BaseEventListener):
    def __init__(self):
        super().__init__()
        self.query_times = []
        self.save_times = []

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(MemoryQueryCompletedEvent)
        def on_memory_query_completed(source, event: MemoryQueryCompletedEvent):
            self.query_times.append(event.query_time_ms)
            avg = sum(self.query_times) / len(self.query_times)
            msg = f"[QUERY ✅] '{event.query}' took {event.query_time_ms:.2f}ms | Avg: {avg:.2f}ms"
            print(msg)
            performance_logger.info(msg)

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_memory_save_completed(source, event: MemorySaveCompletedEvent):
            self.save_times.append(event.save_time_ms)
            avg = sum(self.save_times) / len(self.save_times)
            msg = f"[SAVE ✅] Agent '{event.agent_role}' saved in {event.save_time_ms:.2f}ms | Avg: {avg:.2f}ms"
            print(msg)
            performance_logger.info(msg)


# ✅ Memory Logger 记录查询、保存、检索事件
class MemoryLogger(BaseEventListener):
    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_memory_save_started(source, event: MemorySaveStartedEvent):
            agent = event.agent_role or "Unknown"
            event_logger.info(f"[SAVE ⏳] Agent '{agent}' saving: {event.value[:50]}...")

        @crewai_event_bus.on(MemoryQueryStartedEvent)
        def on_memory_query_started(source, event: MemoryQueryStartedEvent):
            event_logger.info(f"[QUERY ⏳] '{event.query}' started (limit={event.limit})")

        @crewai_event_bus.on(MemoryRetrievalCompletedEvent)
        def on_memory_retrieval_completed(source, event: MemoryRetrievalCompletedEvent):
            task = event.task_id or "-"
            event_logger.info(f"[RETRIEVE ✅] Task={task} | Time={event.retrieval_time_ms:.2f}ms")


# ✅ Memory Error Tracker
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
            msg = f"[SAVE ❌] Memory save failed: {event.error}. {agent_info}"
            print(msg)
            error_logger.error(msg)
            if self.notify_email and self.error_count % 5 == 0:
                self._send_notification(msg)

        @crewai_event_bus.on(MemoryQueryFailedEvent)
        def on_memory_query_failed(source, event: MemoryQueryFailedEvent):
            self.error_count += 1
            msg = f"[QUERY ❌] Memory query failed: {event.error}. Query: '{event.query}'"
            print(msg)
            error_logger.error(msg)
            if self.notify_email and self.error_count % 5 == 0:
                self._send_notification(msg)

    def _send_notification(self, message):
        print(f"[NOTIFICATION] Would send to {self.notify_email}: {message}")

