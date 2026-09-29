# KAVACH Defect Register & Resolution Ledger (Phase 0 Audit)

## Summary
This document tracks all identified software defects, regressions, memory leaks, and synchronization vulnerabilities in the KAVACH repository, along with their resolution status and corresponding regression tests.

---

## Resolved & Active Defect Index

| Defect ID | Severity | Subsystem | Description | Root Cause | Status | Verification Test |
|---|---|---|---|---|---|---|
| **DEF-001** | CRITICAL | cv/__init__.py | ImportError: cannot import name 'ZoneRuleVectorized' | Import referenced non-existent class | **RESOLVED** | 	ests/test_rules.py |
| **DEF-002** | HIGH | cv/rules.py | NameError: name 'travelled' is not defined in NightMovementRule | Variable spelling typo (	raveled vs 	ravelled) | **RESOLVED** | 	ests/test_scenarios.py |
| **DEF-003** | HIGH | core/analytics.py | ImportError: cannot import name '_AsyncStage' in tests | Public/private naming mismatch (_AsyncStage vs AsyncStage) | **RESOLVED** | 	ests/test_startup_state.py |
| **DEF-004** | MEDIUM | setup.py | Windows MSVC compiler flags mapped to unix flags | Platform detection dictionary lookup error on Windows | **RESOLVED** | Build configuration check |
| **DEF-005** | CRITICAL | src/military_core.cpp | parse_output undefined, enqueueV3 setTensorAddress missing | Prototype C++ code incomplete | **RESOLVED** | Native C++20 engine re-architecture |
| **DEF-006** | HIGH | src/military_core.cpp | Static local variables in process_frame causing thread corruption | Multi-camera race condition on shared memory | **RESOLVED** | Instance-scoped buffer allocation |
| **DEF-007** | MEDIUM | src/military_core.cpp | Double-free on ppsink and pipeline destruction | Incorrect GStreamer reference release order | **RESOLVED** | Scoped RAII teardown |
| **DEF-008** | LOW | core/config.py | Duplicate NOTIFY_MIN_SEVERITY definition | Duplicate key in Pydantic settings | **RESOLVED** | Verified clean configuration parsing |
