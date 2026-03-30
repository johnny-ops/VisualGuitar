"""
strum_engine.py
---------------
Tracks the Y-position of the strumming hand over time to detect quick strumming motions.
Returns both trigger and velocity for dynamic playing.
"""

import time
from typing import Tuple

class StrumEngine:
    def __init__(self, velocity_threshold: float = 0.05, cooldown_ms: int = 120):
        """
        velocity_threshold: How much the Y-coordinate must change per frame to trigger
                            a strum. (Normalised 0.0 to 1.0 screen coordinates).
        cooldown_ms:        Minimum time between registered strums to prevent double-trigger.
        """
        self.velocity_threshold = velocity_threshold
        self.cooldown_ms        = cooldown_ms
        self._last_y: float     = 0.0
        self._has_last: bool    = False
        self._last_strum_time   = 0.0
        self._direction: int    = 0  # 1 = down, -1 = up

    def process(self, y_pos: float) -> Tuple[bool, float]:
        """
        Returns (strum_detected, velocity) tuple.
        y_pos: Normalised Y coordinate of the strumming hand (e.g. index finger tip).
        velocity: 0.0 to 1.0 indicating strum strength
        """
        now = time.time()
        strum_detected = False
        velocity = 0.0

        if self._has_last:
            delta_y = y_pos - self._last_y
            abs_velocity = abs(delta_y)
            
            # Detect direction change (strum reversal)
            current_direction = 1 if delta_y > 0 else -1
            direction_changed = (current_direction != self._direction) if self._direction != 0 else False
            
            if abs_velocity > self.velocity_threshold:
                time_since_last = (now - self._last_strum_time) * 1000
                
                # Trigger on direction change OR after cooldown
                if direction_changed or time_since_last > self.cooldown_ms:
                    strum_detected = True
                    self._last_strum_time = now
                    self._direction = current_direction
                    
                    # Map velocity to 0.3-1.0 range (never too quiet)
                    velocity = min(1.0, abs_velocity / 0.15)  # Normalize
                    velocity = 0.3 + (velocity * 0.7)  # Scale to 0.3-1.0

        self._last_y = y_pos
        self._has_last = True
        return strum_detected, velocity
    
    def reset(self):
        """Reset the strum engine state."""
        self._has_last = False
        self._direction = 0
