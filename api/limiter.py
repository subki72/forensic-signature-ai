"""Rate limiter singleton for Forensic Signature AI.

Centralised here to avoid circular imports between main.py and route modules.
Keyed by client IP address; default limit is enforced per-route via decorator.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Global rate limiter — imported by main.py (middleware) and route handlers (decorator)
limiter = Limiter(key_func=get_remote_address)
