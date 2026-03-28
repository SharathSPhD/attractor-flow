"""
Production-grade retry decorator with exponential backoff.

Features:
- Exponential backoff with jitter
- Configurable max_attempts, base_delay, max_delay
- Only retry on specified exception types
- Thread-safe implementation
"""

import functools
import random
import threading
import time
from typing import Callable, Optional, Tuple, Type, TypeVar, Union

# Type variable for generic return type
T = TypeVar('T')

# Thread-local storage for retry state (thread-safe)
_local = threading.local()


class RetryError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None, attempts: int = 0):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


def _calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    jitter: bool = True
) -> float:
    """
    Calculate the delay for the current attempt using exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    # Exponential backoff: base_delay * 2^attempt
    delay = base_delay * (2 ** attempt)

    # Cap at max_delay
    delay = min(delay, max_delay)

    # Add jitter (random value between 0 and delay)
    if jitter:
        delay = delay * random.uniform(0.5, 1.5)

    return delay


# converging: committing to this approach
def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    jitter: bool = True,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exceptions: Exception type(s) to retry on (default: Exception)
        jitter: Add random jitter to delays (default: True)
        on_retry: Optional callback called on each retry with (exception, attempt)

    Returns:
        Decorated function that will retry on specified exceptions

    Example:
        @retry(max_attempts=5, base_delay=0.5, exceptions=(ConnectionError, TimeoutError))
        def fetch_data(url):
            return requests.get(url)

    Thread Safety:
        This decorator is thread-safe. Each thread maintains its own retry state.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")
    if max_delay < base_delay:
        raise ValueError("max_delay must be >= base_delay")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Lock for thread-safe operations on shared state if needed
        _lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Check if this was the last attempt
                    if attempt == max_attempts - 1:
                        raise RetryError(
                            f"Function {func.__name__} failed after {max_attempts} attempts",
                            last_exception=e,
                            attempts=max_attempts
                        ) from e

                    # Calculate delay with exponential backoff
                    delay = _calculate_delay(attempt, base_delay, max_delay, jitter)

                    # Call on_retry callback if provided
                    if on_retry is not None:
                        with _lock:
                            on_retry(e, attempt + 1)

                    # Sleep before next attempt
                    time.sleep(delay)

            # This should never be reached, but for type safety
            raise RetryError(
                f"Function {func.__name__} failed after {max_attempts} attempts",
                last_exception=last_exception,
                attempts=max_attempts
            )

        # Store retry config on the wrapper for introspection
        wrapper.retry_config = {
            'max_attempts': max_attempts,
            'base_delay': base_delay,
            'max_delay': max_delay,
            'exceptions': exceptions,
            'jitter': jitter
        }

        return wrapper

    return decorator


class RetryContext:
    """
    Context manager for retry logic with exponential backoff.

    Thread-safe alternative to the decorator for more complex retry scenarios.

    Example:
        with RetryContext(max_attempts=3, exceptions=(IOError,)) as ctx:
            for attempt in ctx:
                result = risky_operation()
                if result.is_valid:
                    break
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.jitter = jitter
        self._attempt = 0
        self._lock = threading.Lock()
        self._last_exception: Optional[Exception] = None

    def __enter__(self) -> 'RetryContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            return False

        if isinstance(exc_val, self.exceptions):
            with self._lock:
                self._last_exception = exc_val
                self._attempt += 1

                if self._attempt >= self.max_attempts:
                    # Re-raise as RetryError
                    raise RetryError(
                        f"Operation failed after {self.max_attempts} attempts",
                        last_exception=exc_val,
                        attempts=self.max_attempts
                    ) from exc_val

                # Calculate and apply delay
                delay = _calculate_delay(
                    self._attempt - 1,
                    self.base_delay,
                    self.max_delay,
                    self.jitter
                )
                time.sleep(delay)
                return True  # Suppress the exception

        return False  # Don't suppress other exceptions

    def __iter__(self):
        """Allow iteration over attempts."""
        for i in range(self.max_attempts):
            yield i

    @property
    def attempt(self) -> int:
        """Current attempt number (1-indexed)."""
        with self._lock:
            return self._attempt + 1

    @property
    def last_exception(self) -> Optional[Exception]:
        """Last exception that triggered a retry."""
        with self._lock:
            return self._last_exception


# Convenience decorators with common configurations
def retry_on_network_error(max_attempts: int = 5, base_delay: float = 1.0):
    """Retry decorator pre-configured for common network errors."""
    return retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        exceptions=(ConnectionError, TimeoutError, OSError),
        jitter=True
    )


def retry_with_logging(
    logger,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
):
    """Retry decorator that logs each retry attempt."""
    def log_retry(exc: Exception, attempt: int):
        logger.warning(f"Retry attempt {attempt}: {type(exc).__name__}: {exc}")

    return retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        exceptions=exceptions,
        on_retry=log_retry
    )


if __name__ == "__main__":
    # Demo / simple test
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test 1: Basic retry
    call_count = 0

    @retry(max_attempts=3, base_delay=0.1, jitter=False)
    def flaky_function():
        global call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"Attempt {call_count} failed")
        return "Success!"

    result = flaky_function()
    print(f"Test 1 - Basic retry: {result} (took {call_count} attempts)")

    # Test 2: Specific exceptions
    call_count = 0

    @retry(max_attempts=3, base_delay=0.1, exceptions=(ValueError,), jitter=False)
    def specific_exception_function():
        global call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Retryable")
        return "Done"

    result = specific_exception_function()
    print(f"Test 2 - Specific exceptions: {result}")

    # Test 3: RetryError on exhaustion
    @retry(max_attempts=2, base_delay=0.1, jitter=False)
    def always_fails():
        raise RuntimeError("Always fails")

    try:
        always_fails()
    except RetryError as e:
        print(f"Test 3 - RetryError: {e} (attempts: {e.attempts})")

    print("\nAll tests passed!")
