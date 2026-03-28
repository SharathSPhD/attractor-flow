# Use Case 1: Write a Retry Decorator with Exponential Backoff

**Task:** Implement a production-grade `@retry` decorator in Python.

Requirements:
- Exponential backoff with jitter
- Configurable max_attempts, base_delay, max_delay
- Only retry on specified exception types
- Log retry attempts with structured logging
- Thread-safe
- Full test coverage (pytest)

Expected regime trajectory: EXPLORING → CONVERGING
(Agent explores design space, then narrows to implementation)

## Starting point
```python
# retry.py — skeleton only, nothing implemented
def retry(*args, **kwargs):
    pass
```

## Target output
A fully working retry.py + test_retry.py with 8+ passing tests.
