---
name: Feature request
about: Propose a capability or improvement
labels: enhancement
---

**The problem**
What are you trying to do that's hard or impossible today?

**Proposed direction**
What you'd like to see. If it touches the pipeline (synthesize → auto-label → train → run),
say which stage.

**Alternatives considered**
Other approaches, and why this one.

**Scope check**
- [ ] Stays Python-only and Apache-2.0 (no AGPL deps)
- [ ] Keeps the core (`tensor-factory`) CPU-only and dependency-light (GPU/heavy work belongs in a sibling package behind an extra)
