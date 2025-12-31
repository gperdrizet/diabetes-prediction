"""Memory monitoring for batch training with leak detection.

This module provides thread-based memory monitoring to track memory usage
during batch training operations. It captures memory delta from baseline,
samples at configurable intervals, and detects potential memory leaks.
"""

import logging
import threading
import time

import numpy as np
import psutil


class MemoryMonitor:
    """Thread-based memory monitor with leak detection.
    
    Monitors memory usage by capturing a baseline before work begins,
    then sampling memory at regular intervals to calculate delta from
    baseline. After monitoring completes, provides statistics and
    leak detection.
    
    Attributes:
        baseline: Available memory (MB) captured before monitoring starts.
        interval: Sampling interval in seconds (default 0.5s).
        leak_threshold: Memory delta threshold (MB) for leak detection.
    
    Example:
        >>> baseline = psutil.virtual_memory().available / (1024**2)
        >>> monitor = MemoryMonitor(baseline, interval=0.5, leak_threshold_mb=500)
        >>> monitor.start()
        >>> # ... perform work ...
        >>> monitor.stop()
        >>> stats = monitor.get_stats()
        >>> if stats['leak_detected']:
        ...     print(f"Leak: {stats['leak_severity_mb']:.1f}MB")
    """
    
    def __init__(self, baseline_available_mb: float, interval: float = 0.5, 
                 leak_threshold_mb: float = 500.0):
        """Initialize memory monitor.
        
        Args:
            baseline_available_mb: Available system memory in MB before work starts.
                This should be captured immediately before starting monitored work.
            interval: Sampling interval in seconds. Default 0.5s provides good
                balance of detail vs overhead.
            leak_threshold_mb: Memory delta threshold for leak detection. If final
                memory delta exceeds this value, a leak is flagged. Default 500MB.
        """
        self.baseline = baseline_available_mb
        self.interval = interval
        self.leak_threshold = leak_threshold_mb
        
        self._samples: list[float] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float | None = None
    
    def start(self) -> None:
        """Start memory monitoring in background thread.
        
        Creates a daemon thread that samples memory at the configured
        interval until stop() is called.
        """
        self._start_time = time.time()
        self._stop_event.clear()
        self._samples = []  # Reset samples for new monitoring session
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def _monitor_loop(self) -> None:
        """Background thread loop that samples memory at configured interval."""
        while not self._stop_event.is_set():
            # Get current available memory
            current_available = psutil.virtual_memory().available / (1024**2)
            # Calculate delta: positive = memory consumed since baseline
            delta = self.baseline - current_available
            
            with self._lock:
                self._samples.append(delta)
            
            time.sleep(self.interval)
    
    def stop(self) -> None:
        """Stop memory monitoring.
        
        Signals the monitoring thread to stop and waits for it to exit.
        Logs a warning if thread doesn't exit within 2 second timeout.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logging.warning('Memory monitor thread did not exit within 2s timeout')
    
    def get_stats(self) -> dict:
        """Get memory statistics and leak detection results.
        
        Returns:
            Dictionary containing:
                - max_delta_mb: Maximum memory delta observed (float)
                - mean_delta_mb: Average memory delta (float)
                - n_samples: Number of samples collected (int)
                - duration_sec: Monitoring duration in seconds (float)
                - leak_detected: Whether a leak was detected (bool)
                - leak_severity_mb: Size of leak if detected, else 0.0 (float)
        """
        with self._lock:
            if not self._samples:
                return {
                    'max_delta_mb': 0.0,
                    'mean_delta_mb': 0.0,
                    'n_samples': 0,
                    'duration_sec': 0.0,
                    'leak_detected': False,
                    'leak_severity_mb': 0.0
                }
            
            samples_array = np.array(self._samples)
            final_delta = self._samples[-1]
            
            # Leak detection: if final delta exceeds threshold, memory wasn't released
            leak_detected = final_delta > self.leak_threshold
            
            return {
                'max_delta_mb': float(np.max(samples_array)),
                'mean_delta_mb': float(np.mean(samples_array)),
                'n_samples': len(self._samples),
                'duration_sec': time.time() - self._start_time if self._start_time else 0.0,
                'leak_detected': leak_detected,
                'leak_severity_mb': float(final_delta) if leak_detected else 0.0
            }
