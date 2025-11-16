# Security Audit - Action Items

This document outlines the actionable items identified in the security audit conducted on 2025-11-16.

## High Priority Issues

### H-1: Monkey Patching of AIMessageChunk

**Status:** 🔴 **REQUIRES IMMEDIATE ATTENTION**

**Files Affected:**
- `langchain_qwq/chat_models.py:247`
- `langchain_qwq/chat_models.py:508`

**Issue:**
The code modifies `AIMessageChunk.__add__` globally, which can affect all instances across the entire application.

**Current Code:**
```python
# Monkey patch the __add__ method
AIMessageChunk.__add__ = patched_add  # type: ignore

try:
    # ... streaming code ...
finally:
    # Restore the original method
    AIMessageChunk.__add__ = original_add  # type: ignore
```

**Recommended Solutions:**

**Option 1: Use a Custom Subclass (RECOMMENDED)**
```python
class QwQMessageChunk(AIMessageChunk):
    """Custom message chunk that preserves tool calls during addition."""

    def __add__(self, other: AIMessageChunk) -> AIMessageChunk:
        result = super().__add__(other)

        # Custom tool call preservation logic here
        if hasattr(self, "tool_calls") and self.tool_calls:
            # ... preservation logic ...

        return result
```

**Option 2: Context Manager with Thread-Local Storage**
```python
import threading
from contextlib import contextmanager

_original_add = AIMessageChunk.__add__
_patched_threads = threading.local()

@contextmanager
def patched_message_chunk():
    """Thread-safe context manager for patching."""
    thread_id = threading.get_ident()

    if not hasattr(_patched_threads, 'active'):
        _patched_threads.active = set()

    _patched_threads.active.add(thread_id)
    AIMessageChunk.__add__ = patched_add

    try:
        yield
    finally:
        _patched_threads.active.discard(thread_id)
        if not _patched_threads.active:
            AIMessageChunk.__add__ = _original_add
```

**Option 3: Composition Pattern**
```python
class ToolCallPreservingStream:
    """Wrapper that preserves tool calls without modifying global state."""

    def __init__(self, stream):
        self.stream = stream
        self.accumulated_tool_calls = []

    def __iter__(self):
        for chunk in self.stream:
            # Custom logic to preserve tool calls
            chunk = self._preserve_tool_calls(chunk)
            yield chunk

    def _preserve_tool_calls(self, chunk):
        # Implementation without global modifications
        pass
```

**Timeline:** Within 1 week

---

## Medium Priority Issues

### M-1: Error Message Sanitization

**Status:** 🟡 **NEEDS IMPROVEMENT**

**Files Affected:**
- `langchain_qwq/base.py:64-70`
- Multiple error handling locations

**Recommended Fix:**
```python
# Add custom exception class with automatic sanitization
class QwQException(Exception):
    """Base exception with automatic secret sanitization."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        # Sanitize message to remove any potential secrets
        sanitized_message = self._sanitize_message(message)
        super().__init__(sanitized_message)
        self.original_error = original_error

    @staticmethod
    def _sanitize_message(message: str) -> str:
        """Remove potential API keys and sensitive data from message."""
        import re
        # Redact anything that looks like an API key
        message = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED]', message)
        message = re.sub(r'DASHSCOPE_API_KEY[=:]\s*\S+', 'DASHSCOPE_API_KEY=[REDACTED]', message)
        return message

# Usage in code:
if not self.api_key:
    raise QwQException("API key is required but not provided")
```

**Timeline:** Within 2 weeks

---

### M-2: Input Validation

**Status:** 🟡 **NEEDS IMPROVEMENT**

**Recommended Additions:**

**1. Message Length Validation:**
```python
# In base.py or chat_models.py
MAX_MESSAGE_LENGTH = 100000  # Adjust based on API limits
MAX_MESSAGES_PER_REQUEST = 100

def _validate_messages(self, messages: List[BaseMessage]) -> None:
    """Validate message inputs before sending to API."""
    if len(messages) > MAX_MESSAGES_PER_REQUEST:
        raise ValueError(f"Too many messages: {len(messages)} > {MAX_MESSAGES_PER_REQUEST}")

    total_length = sum(len(str(msg.content)) for msg in messages)
    if total_length > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Total message length {total_length} exceeds maximum {MAX_MESSAGE_LENGTH}")
```

**2. Thinking Budget Validation:**
```python
# In chat_models.py ChatQwen class
from pydantic import field_validator

@field_validator('thinking_budget')
@classmethod
def validate_thinking_budget(cls, v: Optional[int]) -> Optional[int]:
    if v is not None:
        if v < 1:
            raise ValueError("thinking_budget must be at least 1")
        if v > 10000:  # Set reasonable upper limit
            raise ValueError("thinking_budget too large, maximum is 10000")
    return v
```

**3. Model Name Validation:**
```python
@field_validator('model_name')
@classmethod
def validate_model_name(cls, v: str) -> str:
    if len(v) > 100:
        raise ValueError("model_name too long")
    if not re.match(r'^[a-zA-Z0-9._-]+$', v):
        raise ValueError("model_name contains invalid characters")
    return v
```

**Timeline:** Within 3 weeks

---

### M-3: JSON Parsing Safety

**Status:** 🟡 **NEEDS IMPROVEMENT**

**Recommended Fix:**
```python
# In chat_models.py
import logging

logger = logging.getLogger(__name__)

def _safe_json_parse(self, json_string: str, strict_mode: bool = False) -> dict:
    """Parse JSON with optional repair and logging."""
    try:
        # Try standard parsing first
        return json.loads(json_string, strict=True)
    except json.JSONDecodeError as e:
        if strict_mode:
            raise

        # Log when repair is needed
        logger.warning(
            "JSON repair needed for malformed API response",
            extra={"error_position": e.pos}
        )

        try:
            # Use json-repair as fallback
            import json_repair
            repaired = json_repair.loads(json_string)
            logger.info("JSON successfully repaired")
            return repaired
        except Exception as repair_error:
            logger.error("JSON repair failed", exc_info=True)
            raise ValueError("Unable to parse API response") from repair_error

# Add configuration option
class ChatQwQ(_BaseChatQwen):
    strict_json: bool = Field(default=False)
    """If True, disable automatic JSON repair"""
```

**Timeline:** Within 3 weeks

---

### M-4: Error Information Disclosure

**Status:** 🟡 **NEEDS IMPROVEMENT**

**Recommended Fix:**
```python
# Create sanitized error messages
class APIResponseError(QwQException):
    """Raised when API returns invalid response."""
    pass

# In error handling:
try:
    # ... parsing code ...
except JSONDecodeError as e:
    # Log detailed error internally
    logger.error(
        "API response parsing failed",
        extra={
            "error": str(e),
            "position": e.pos,
            "model": self.model_name
        }
    )

    # Raise sanitized error to user
    raise APIResponseError(
        "Invalid API response received. Please try again or contact support if the issue persists."
    ) from None  # Suppress exception chain
```

**Timeline:** Within 3 weeks

---

## Low Priority Issues

### L-1: Rate Limiting

**Status:** 🟢 **ENHANCEMENT**

**Recommended Addition:**
```python
# Add to documentation and consider implementing:
"""
Rate Limiting Best Practices:

For production deployments, implement rate limiting at the application level:

```python
from functools import wraps
from time import time, sleep
from threading import Lock

class RateLimiter:
    def __init__(self, calls_per_second: float = 10):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self.lock = Lock()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                elapsed = time() - self.last_call
                if elapsed < self.min_interval:
                    sleep(self.min_interval - elapsed)
                self.last_call = time()
            return func(*args, **kwargs)
        return wrapper

# Usage:
model = ChatQwQ(model="qwq-plus")
rate_limiter = RateLimiter(calls_per_second=10)

@rate_limiter
def safe_invoke(prompt):
    return model.invoke(prompt)
```
"""
```

**Timeline:** Documentation update within 1 month

---

### L-2: Timeout Validation

**Status:** 🟢 **ENHANCEMENT**

**Recommended Fix:**
```python
from pydantic import field_validator

class _BaseChatQwen(BaseChatOpenAI):
    @field_validator('request_timeout')
    @classmethod
    def validate_timeout(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if v < 1:
                raise ValueError("timeout must be at least 1 second")
            if v > 300:  # 5 minutes
                raise ValueError("timeout cannot exceed 300 seconds")
        return v
```

**Timeline:** Within 1 month

---

### L-3: Regex Safety

**Status:** 🟢 **ENHANCEMENT**

**Recommended Fix:**
```python
def _is_open_source_model(self) -> bool:
    """Check if model is open source based on naming convention."""
    # Add length check before regex
    if len(self.model_name) > 100:
        return False

    # Use simpler string operation instead of regex
    model_lower = self.model_name.lower()

    # Check for common patterns like "32b", "72b", etc.
    import re
    pattern = r'\d+b'

    # Set timeout for regex (Python 3.11+) or use simpler check
    try:
        return bool(re.search(pattern, model_lower))
    except re.error:
        # Fallback to simple string check
        return 'b' in model_lower and any(c.isdigit() for c in model_lower)
```

**Timeline:** Within 1 month

---

## Informational Items

### I-1: Fix Typos

**Files to Update:**
- `langchain_qwq/chat_models.py` (multiple locations)

**Change:** "Thingking" → "Thinking"

**Search and replace:**
```bash
find . -type f -name "*.py" -exec sed -i 's/Thingking/Thinking/g' {} +
```

**Timeline:** Immediate

---

### I-2: Add Security Documentation

**Files to Create/Update:**
- ✅ `SECURITY.md` (created)
- `README.md` (add security section)
- `docs/security-best-practices.md` (new file)

**README.md Addition:**
```markdown
## Security

### Reporting Security Issues

Please see [SECURITY.md](SECURITY.md) for information on reporting security vulnerabilities.

### Security Best Practices

1. **Never commit API keys** - Use environment variables
2. **Validate all inputs** - Implement input validation at the application level
3. **Use timeouts** - Set reasonable timeout values
4. **Monitor usage** - Implement logging and monitoring
5. **Keep updated** - Regularly update dependencies

For detailed security guidance, see [SECURITY.md](SECURITY.md).
```

**Timeline:** Within 1 week

---

## Dependency Updates

### Required Updates (System Dependencies)

These are not direct dependencies but should be updated in deployment environments:

```bash
# Update system packages
pip install --upgrade pip setuptools

# If cryptography is a transitive dependency, ensure it's updated:
pip install --upgrade "cryptography>=43.0.1"
```

**Timeline:** Immediate

---

## CI/CD Integration

### Add Security Scanning to CI Pipeline

**Create `.github/workflows/security.yml`:**
```yaml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly scan

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Run pip-audit
        run: |
          pip install pip-audit
          pip-audit

      - name: Run bandit
        run: |
          pip install bandit
          bandit -r langchain_qwq/

      - name: Run safety
        run: |
          pip install safety
          safety check
```

**Timeline:** Within 2 weeks

---

## Summary Checklist

### Immediate (Week 1)
- [ ] Fix typo: "Thingking" → "Thinking"
- [ ] Review and plan monkey-patching refactor
- [ ] Add SECURITY.md to repository ✅ (completed)
- [ ] Update README with security section

### Short-term (Weeks 2-4)
- [ ] Implement solution for H-1 (monkey patching)
- [ ] Add error message sanitization (M-1)
- [ ] Implement input validation (M-2)
- [ ] Add JSON parsing safety with logging (M-3)
- [ ] Sanitize error disclosure (M-4)
- [ ] Set up CI security scanning

### Medium-term (Months 2-3)
- [ ] Add timeout validation (L-2)
- [ ] Improve regex safety (L-3)
- [ ] Create security best practices documentation
- [ ] Add security-focused test cases
- [ ] Implement rate limiting examples

### Ongoing
- [ ] Monitor dependencies for new vulnerabilities
- [ ] Regular security audits (quarterly)
- [ ] Update security documentation
- [ ] Respond to security reports

---

**Created:** 2025-11-16
**Last Updated:** 2025-11-16
**Next Review:** 2025-02-16
