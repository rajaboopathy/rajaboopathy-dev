---
title: "Synthesis: Designing Data-Intensive Applications"
description: "Key takeaways and systems architecture principles synthesized from Martin Kleppmann's seminal book."
pubDate: 2026-08-05
category: "Book Notes"
tags: ["Distributed Systems", "Databases", "Systems Design"]
---

Martin Kleppmann’s *Designing Data-Intensive Applications* is the gold standard for understanding distributed systems and database internals. Here is a high-level synthesis of its key concepts.

## 1. Reliability, Scalability, and Maintainability
- **Reliability**: Systems should continue to work correctly even when things go wrong (tolerating hardware, software, and human errors).
- **Scalability**: Having strategies to cope with growth (e.g., separating reads and writes, horizontal partitioning).
- **Maintainability**: Building systems so that new engineers can understand and modify them easily.

## 2. Storage Engines: LSM-Trees vs. B-Trees
- **LSM-Trees (Log-Structured Merge-Trees)**: Optimized for write-heavy workloads. Writes are appended to an in-memory memtable, flushed to SSTables on disk, and periodically compacted. (Examples: Cassandra, RocksDB).
- **B-Trees**: Optimized for read-heavy workloads. They structure disk space into fixed-size pages and maintain a balanced tree structure. (Examples: PostgreSQL, MySQL).

## 3. The Fallacy of Distributed Transactions
In distributed systems, achieving consensus (e.g., 2-Phase Commit) is slow and prone to blocking. Modern systems often favor:
- **Eventual Consistency**: Replicas will eventually catch up, but might read stale data temporarily.
- **Saga Pattern**: A sequence of local transactions, where each transaction updates the database and triggers the next step or compensates on failure.

This book teaches us to look past marketing buzzwords and evaluate data systems based on their trade-offs.
