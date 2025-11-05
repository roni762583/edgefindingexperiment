# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.


# CLAUDE.md - STRICT PRODUCTION REQUIREMENTS

## 🚨 CRITICAL: MISSION-CRITICAL PRODUCTION SYSTEM

This is a **PRODUCTION-GRADE, MISSION-CRITICAL** trading system. There is **ZERO TOLERANCE** for shortcuts, assumptions, or incomplete implementations.

## 🏦 OANDA LIVE TRADING ENVIRONMENT

**MANDATORY**: This system uses **LIVE OANDA trading accounts** with **REAL MONEY**. 

### OANDA Configuration Requirements:
- **Environment**: `OANDA_ENVIRONMENT=live` (NOT practice)
- **API Endpoints**: Live trading endpoints only (api-fxtrade.oanda.com)
- **Official v20 Library**: Use ONLY the official OANDA v20 Python library (v20>=3.0.25.0)
- **Rate Limits**: Respect production rate limits for live accounts
- **Error Handling**: Fail fast on any API errors - no retries that could cause financial loss

### Live Trading Safety:
- **Data Download Only**: Initially used for historical data retrieval
- **No Automated Trading**: Trading decisions require explicit human authorization
- **Position Monitoring**: All positions must be tracked and logged
- **Risk Management**: Hard stops and position size limits mandatory

## ⚡ ABSOLUTE REQUIREMENTS
### 0. AVOID WRITTING NEW SCRIPTS, FIX/DEBUG EXISTING ONES!
SCRIPT CREATION POLICY - STRICT RULES

    AVOID CREATING NEW SCRIPTS - Always fix/debug/improve existing scripts first
    IF A NEW SCRIPT IS ABSOLUTELY NECESSARY:
        Temporary scripts (testing/debugging/one-off operations):
            MUST be placed in /tmp/ directory (create if needed)
            Name with descriptive prefix: temp_[purpose]_[timestamp].py
        Permanent scripts (essential project functionality):
            MUST inform user and get explicit confirmation BEFORE creation
            Explain WHY existing scripts cannot be modified
            Show exact location and purpose

### 1. FOLLOW INSTRUCTIONS EXACTLY
- **DO NOT** assume what the user wants
- **DO NOT** add features not explicitly requested
- **DO NOT** skip steps or take shortcuts
- Execute **EXACTLY** what is requested - nothing more, nothing less

### 2. VERIFICATION IS MANDATORY
- **NEVER** report completion without verification
- **ALWAYS** test and confirm functionality before claiming completion
- **ALWAYS** show concrete evidence of verification (output, test results, etc.)
- If you cannot verify, explicitly state "UNVERIFIED" in your response

### 3. CODE QUALITY STANDARDS
- **NO MONKEY PATCHES** - ever
- **NO STUB FUNCTIONS** - implement fully or not at all
- **NO PLACEHOLDER CODE** - every line must be production-ready
- **NO APPROXIMATIONS** - exact implementations only
- **NO SILENT FALLBACKS** - fail fast and loud

### 4. PRODUCTION CODE REQUIREMENTS
- **Type hints** on every function
- **Docstrings** with clear specifications
- **Error handling** that fails explicitly (no silent catches)
- **Input validation** on all boundaries
- **Logging** for critical operations
- **No magic numbers** - use named constants
- **DRY principle** - no code duplication
- **Single responsibility** - each function does ONE thing

### 5. FAILURE PHILOSOPHY
```python
# ❌ WRONG - Silent fallback
try:
    result = complex_calculation()
except:
    result = 0  # NO! Never approximate on failure

# ✅ CORRECT - Fail explicitly
try:
    result = complex_calculation()
except SpecificException as e:
    logger.error(f"Calculation failed: {e}")
    raise  # Let it crash - no silent failures
```

### 6. REPORTING STANDARDS
When reporting task completion:
1. State **EXACTLY** what was done
2. Show **VERIFICATION** output
3. List any **ASSUMPTIONS** made (should be zero)
4. Highlight any **DEVIATIONS** from request
5. Provide **TEST RESULTS** when applicable

### 7. IMPLEMENTATION CHECKLIST
Before claiming any task is complete:
- [ ] Code runs without errors
- [ ] All edge cases handled
- [ ] No TODOs or FIXMEs remain
- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] Error handling is explicit
- [ ] Code follows DRY principle
- [ ] No magic numbers
- [ ] Verified with actual execution

### 8. RED FLAGS - NEVER DO THESE

1. **"This should work"** - Test it or don't claim it
2. **"Approximately correct"** - Exact or nothing
3. **"You can add X later"** - Complete it now
4. **"This is a simple implementation"** - Make it production-ready
5. **"Fallback to default"** - Crash instead
6. **Silent exceptions** - All failures must be loud
7. **Untested code** - If you didn't run it, it's not done

## Always respond with concise, professional clarity.
## Explicitly outline steps or decisions.

## ✅ CORRECT APPROACH EXAMPLE

User: "Calculate the moving average"

❌ WRONG:
```python
def moving_avg(data, window):
    # Simple implementation
    return sum(data[-window:]) / window
```

✅ CORRECT:
```python
from typing import List, Union
import numpy as np

def calculate_moving_average(
    data: np.ndarray, 
    window: int,
    min_periods: int = 1
) -> np.ndarray:
    """
    Calculate simple moving average with proper handling.
    
    Args:
        data: Input array of numerical values
        window: Window size for moving average
        min_periods: Minimum number of observations required
        
    Returns:
        Array of moving averages
        
    Raises:
        ValueError: If window <= 0 or window > len(data)
        TypeError: If data is not numeric
    """
    if window <= 0:
        raise ValueError(f"Window must be positive, got {window}")
    if window > len(data):
        raise ValueError(f"Window {window} exceeds data length {len(data)}")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"Data must be numeric, got {data.dtype}")
        
    # Use pandas for correct edge handling
    import pandas as pd
    return pd.Series(data).rolling(
        window=window, 
        min_periods=min_periods
    ).mean().values
```

## 🔍 VERIFICATION TEMPLATE

After completing any task:

```
TASK COMPLETED: [Exact description]

VERIFICATION:
- Executed: [command/function]
- Output: [actual output]
- Tests passed: [list specific tests]
- Edge cases verified: [list cases]

IMPLEMENTATION DETAILS:
- Files modified: [list files]
- Functions added: [list with signatures]
- Error handling: [describe approach]

DEVIATIONS: None [or list any]
```

## ⚠️ REMEMBER

**This is a PRODUCTION TRADING SYSTEM handling REAL MONEY. Every line of code matters. Every assumption is a potential financial loss. Every shortcut is a system failure waiting to happen.**

**FAIL FAST. FAIL LOUD. NEVER APPROXIMATE.**

---

## 🔒 PROTECTED CONTENT ABOVE - DO NOT MODIFY

**⚠️ CRITICAL**: Everything above this line is IMMUTABLE. Any modifications, additions, or updates MUST be appended below this section.

---
