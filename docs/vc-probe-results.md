# VirtualClient Eval Path Probe Results

**Date:** 2026-03-06
**Host:** M3 Max, macOS 15+
**Branch:** `feat/vc-eval-probe`

## Class Discovery (all present)

| Class | Found | Methods |
|-------|-------|---------|
| `_ANEVirtualClient` | YES | 22 class + 66 instance |
| `_ANESharedEvents` | YES | 2 class + 8 instance |
| `_ANESharedWaitEvent` | YES | 3 class + 8 instance |
| `_ANESharedSignalEvent` | YES | (from chaining probe) |
| `IOSurfaceSharedEvent` | YES | +new returns nil |

## Selector/Capability Discovery

| Capability | Available |
|-----------|-----------|
| `_ANEClient.virtualClient` (property) | YES |
| `_ANEVirtualClient.evaluateWithModel:options:request:qos:error:` | YES |
| `_ANEVirtualClient.doEvaluateWithModel:...completionEvent:...` | YES |
| `_ANEVirtualClient.doMapIOSurfacesWithModel:...` | YES |
| `_ANEVirtualClient.loadModel:options:qos:error:` | YES |
| `_ANERequest.setSharedEvents:` | YES |
| `_ANERequest.setCompletionHandler:` | YES |
| 8-arg request factory with `sharedEvents:` | NO |

## Critical Finding: virtualClient Returns nil

**All eval paths blocked at stage 1 (NO_VIRTUAL_CLIENT).**

The `_ANEClient` obtained via `_ANEInMemoryModel.sharedConnection` returns nil for the `virtualClient` property. This is the fundamental blocker.

**Implication:** The `virtualClient` property likely requires a different `_ANEClient` instantiation path — possibly:
1. Direct `[[_ANEClient alloc] init]` or a factory method
2. A specific connection type or entitlement
3. The property is populated only for file-based `_ANEModel` (not in-memory)

## IOSurfaceSharedEvent: +new Returns nil

Confirmed golden output constraint: `IOSurfaceSharedEvent +new` returns nil. Factory methods are required (likely obtained from a `MTLSharedEvent` or similar Metal object).

## Timing (non-findings)

All eval probes exit at stage 1 before reaching any eval call, so no timing data was captured.

## Next Steps

1. **Probe `_ANEClient` instantiation:** Try `[[_ANEClient alloc] init]` directly, or probe class methods for alternative factories
2. **Probe file-based `_ANEModel` path:** The virtualClient may work with `_ANEModel` (file-based) rather than `_ANEInMemoryModel`
3. **Probe `_ANEClient.sharedConnection` on the class directly:** Instead of going through `_ANEInMemoryModel.sharedConnection`, try `[_ANEClient sharedConnection]` as a class method
4. **Metal-backed IOSurfaceSharedEvent:** Create via `MTLDevice.makeSharedEvent()` and wrap
5. **If virtualClient remains inaccessible:** Pivot to compute-side decode improvements (MIL reformulation to reduce kernel count)
