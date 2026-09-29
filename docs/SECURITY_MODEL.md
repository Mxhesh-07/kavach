# KAVACH Security Model & Integrity Safeguards

## 1. Threat Vectors & Mitigations
- **Video Feed Tampering / Man-in-the-Middle**: RTSPS (TLS 1.3) ingestion encryption and SHA-256 frame hash chaining (`core/hashchain.py`).
- **Database & Alert Log Modification**: SQLite WAL mode with append-only HMAC hashing on each recorded alert event.
- **Buffer Overflows & Memory Safety**: C++20 native modules use RAII, `std::span`, and bounded zero-allocation fixed ring buffers to eliminate arbitrary pointer arithmetic vulnerabilities.
