# Engineering Design Decisions

This document records architectural and implementation decisions for the Medicare RAG assignment.

The goal is to make important choices traceable and explainable during code review.

Each decision has a status:

- **Accepted** — implemented and supported by inspection, tests, or measured evidence.
- **In Progress** — implementation exists but validation is still being completed.
- **Planned** — intended approach; not yet implemented or validated.
- **Rejected** — evaluated and intentionally not used.

A planned decision must not be presented as implemented until its supporting code and validation exist.

---

## Decision 001 — PDF extraction order

**Status:** Accepted — validated during PDF inspection

### Decision

Use PyMuPDF's native extraction order together with positional/layout metadata.

Do not globally use:

```python
page.get_text("text", sort=True)

as the canonical document representation.

Evidence

Inspection of physical PDF pages 10 and 11 showed two-column comparison layouts.

Using sort=True interleaved text from the Original Medicare and Medicare Advantage columns, reducing semantic coherence.

Native extraction order preserved the logical column grouping more effectively.

Trade-off

Native extraction order is still PDF-dependent and is not guaranteed to produce correct semantic ordering for every possible PDF.

Therefore, positional metadata such as bounding boxes, block indexes, and line indexes is preserved for structural reconstruction.

Implementation

app/rag/pdf_parser.py
------------------------------------------------------

Decision 002 — OCR is not required

Status: Accepted — validated during PDF inspection

Decision

Do not add OCR to the ingestion pipeline.

Evidence

The supplied Medicare PDF is text-native.

PyMuPDF detected:

128 physical PDF pages.
Approximately 346,000 extracted text characters.
A median of approximately 2,778 extracted characters per page.

Only physical pages 1, 118, and 127 contained fewer than 100 extracted characters.

Manual inspection showed:

Page 1 is the cover.
Page 118 is a notes page.
Page 127 contains no meaningful text content.

These low-text pages therefore do not indicate OCR failure.

Trade-off

The parser is intentionally optimized for the supplied assignment PDF rather than scanned-image PDFs.

If support for scanned documents were required in a future production system, OCR could be added as a separate ingestion capability.

Implementation

app/rag/pdf_parser.py

scripts/inspect_pdf.py

------------------------------------------------------------------
Decision 003 — Infer body font statistically

Status: Accepted — implemented and tested

Decision

Infer the dominant body font size from the supplied document instead of hardcoding a font size such as 11 pt.

Evidence

Document inspection showed that 11 pt text overwhelmingly dominates the handbook.

However, hardcoding that observation would unnecessarily couple the parser to one document revision.

The parser therefore determines the dominant font size using character-weighted font-size frequency.

Trade-off

Font size alone cannot reliably determine semantic structure.

It is used as one structural signal together with boldness, text length, and layout information.

Implementation

infer_body_font_size() in:

app/rag/pdf_parser.py

Validation

Covered by:

tests/test_pdf_parser.py

----------------------------------------------------------------
Decision 004 — Parse at line/span level instead of block level

Status: Accepted — implemented and tested

Decision

Use PyMuPDF line/span-level extraction metadata rather than treating each PDF block as one semantic paragraph.

Evidence

Inspection showed that individual PyMuPDF blocks may contain multiple semantic elements.

Examples include:

headings followed by body text in the same block;
multiple bullet items inside the same block;
enrollment-period headings embedded with explanatory content.

Treating a complete block as one paragraph would therefore lose useful document structure.

Trade-off

Line/span processing requires additional reconstruction logic.

That complexity is justified because it improves heading, paragraph, and list-item separation before chunking.

Implementation

extract_page_lines()

reconstruct_units()

in:

app/rag/pdf_parser.py

----------------------------------------------------------------
## Decision 005 — Remove non-semantic page boilerplate before retrieval

**Status:** Accepted — implemented and manually validated

### Decision

Remove standalone page numbers and repeated header/footer boilerplate before semantic chunk construction and embedding.

### Evidence

PDF inspection showed repeated page-number and running-header artifacts near physical page boundaries.

These elements are useful for document presentation but introduce noise when repeated across retrieval chunks.

### Approach

The parser:

1. Detects standalone page-number lines.
2. Examines header and footer regions.
3. Detects repeated boilerplate across multiple physical pages.
4. Excludes matching lines from semantic document units.

### Trade-off

Aggressive boilerplate removal could accidentally remove meaningful repeated content.

Filtering is therefore restricted to standalone page-number patterns and repeated text located within defined header/footer regions.

### Implementation

`find_repeated_boilerplate()`

`is_boilerplate_line()`

in:

`app/rag/pdf_parser.py`

### Validation

Automated parser tests verify standalone page-number removal.

Manual inspection confirmed that:

- page-number artifacts are removed from semantic units;
- repeated running headers are excluded;
- physical page 118 retains only the meaningful `Notes` heading;
- blank physical page 127 produces no semantic units.

-----------------------------------------------------------------------
Decision 006 — Use exact FAISS retrieval for the initial vector index

Status: Planned — retrieval phase not yet implemented

Proposed Decision

Use normalized embeddings with FAISS exact inner-product search as the initial retrieval baseline.

Rationale

The assignment contains one static PDF and is expected to produce only hundreds or low thousands of chunks.

Approximate nearest-neighbor infrastructure would add unnecessary complexity at this scale.

Validation Required

This decision will be accepted only after:

the final chunk corpus is built;
the embedding model is tested;
retrieval metrics are measured.
Proposed Implementation

app/rag/vector_store.py

--------------------------------------------------------------------
## Decision 007 — Derive chunk candidates from document statistics

**Status:** In Progress — document-derived candidate generation and
structure-aware chunk construction implemented; retrieval evaluation pending

### Decision

Do not use a user-defined or arbitrarily hardcoded chunk size.

Candidate chunk targets are derived automatically from measured characteristics
of the supplied document, including:

- semantic-unit token distributions;
- paragraph token distributions;
- reconstructed structural-section distributions;
- percentile statistics;
- embedding-model token constraints.

All derived candidates are processed using the same structure-aware chunking
algorithm. Their retrieval performance will be evaluated empirically before a
final strategy is selected.

### Evidence

Token profiling with the tokenizer associated with
`BAAI/bge-small-en-v1.5` produced:

- 2,056 semantic units;
- maximum semantic-unit length of 245 tokens;
- no semantic unit exceeding the 512-token model limit;
- 453 reconstructed structural sections;
- section median of 100 tokens;
- section P75 of 187 tokens;
- section P90 of 331.8 tokens;
- section P95 of 412.2 tokens;
- 16 structural sections exceeding the 512-token model limit;
- maximum structural-section length of 1,396 tokens.

These measurements indicate that normal semantic units can generally remain
intact while large structural sections must be divided at semantic-unit
boundaries.

### Structural Section Reconstruction

Consecutive heading lines on the same page are kept together so that visually
split headings remain one logical heading.

A heading appearing on a new physical page starts a new structural section
when the preceding section contains only heading material. This prevents
unrelated page-level headings from being merged simply because no paragraph
occurred between them.

This refinement was identified through manual inspection of the supplied
Medicare document.

### Candidate Derivation

Candidate targets are generated from document statistics rather than supplied
by the user or stored as a fixed list.

The derivation uses:

- the greater of section median and paragraph P95;
- section P75;
- section P90;
- section P95.

Raw values are normalized to the nearest practical 32-token boundary,
deduplicated, sorted, and capped by the embedding model's maximum token length.

For the supplied Medicare document, this produces:

`128, 192, 320, 416`

These values are outputs of the derivation logic, not hardcoded chunk-size
configuration.

### Candidate Corpus Construction

Each candidate target is applied using the same structure-aware chunking
algorithm so that subsequent evaluation compares chunk size rather than
different chunking algorithms.

The candidate target is treated as a soft packing objective rather than a
destructive boundary.

Complete paragraphs and list items are preserved whenever they fit within the
embedding model's hard token limit. Oversized structural sections are divided
by greedily packing complete semantic units.

Leading heading context is retained for continuation chunks.

The embedding model's 512-token maximum is treated as a hard constraint.

For the supplied Medicare document, candidate construction produced:

| Target | Chunks | Mean tokens | Median tokens | Max tokens | Above soft target |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 798 | 86.82 | 92 | 260 | 53 |
| 192 | 632 | 107.11 | 109 | 260 | 14 |
| 320 | 523 | 127.16 | 106 | 320 | 0 |
| 416 | 481 | 137.37 | 106 | 411 | 0 |

Chunks exceeding the 128- or 192-token soft target occur when preserving an
entire semantic unit is preferable to splitting it solely to meet the target.

No generated chunk exceeds the 512-token hard limit.

### Rationale

Dynamic chunk-size selection is a primary requirement of the assignment.

Using a conventional fixed value such as 256 or 512 tokens and labeling it
dynamic would not satisfy that requirement.

Deriving candidate sizes from measured document characteristics makes the
process document-adaptive, reproducible, and explainable.

However, document statistics alone do not determine the final winner.
Retrieval quality will determine which candidate strategy is ultimately
selected.

### Validation Completed

The current implementation verifies that:

- consecutive heading lines on the same page remain together;
- headings crossing into a new physical page are separated when appropriate;
- candidate targets are derived from supplied document statistics;
- values are normalized to practical token boundaries;
- duplicate targets are removed;
- targets respect embedding-model constraints;
- semantic units are preserved rather than unnecessarily split;
- continuation chunks retain heading context;
- chunk identifiers are deterministic and unique;
- source-unit coverage is preserved;
- page provenance is retained;
- all candidate corpora contain non-empty chunks;
- no generated chunk exceeds the 512-token hard limit.

### Validation Remaining

Candidate strategies still require empirical retrieval evaluation.

Planned metrics include:

- Recall@K;
- Mean Reciprocal Rank (MRR);
- NDCG@K;
- semantic-boundary preservation;
- chunk-length efficiency;
- lightweight semantic coherence if it provides useful discriminative signal.

The final chunk strategy will be selected only after retrieval evaluation.

### Implementation

Implemented:

- `app/rag/chunking.py`
- `app/rag/tokenization.py`
- `scripts/profile_token_lengths.py`
- `scripts/build_candidate_chunks.py`
- `tests/test_tokenization.py`
- `tests/test_chunking.py`

Generated evidence:

- `artifacts/token_profile.json`
- `artifacts/chunk_strategy_profile.json`

Generated candidate corpora under `artifacts/chunks/` are reproducible build
artifacts and are not committed to source control.

Planned:

- `app/rag/chunk_evaluation.py`

-----------------------------------------------------------------
Decision 008 — Backend owns citation and confidence metadata

Status: Planned — generation phase not yet implemented

Proposed Decision

The LLM should generate only information it must reason about, primarily:

grounded answer text;
citation IDs selected from an explicit allow-list of retrieved sources.

The backend should own:

physical page numbers;
chunk IDs;
retrieval scores;
chunking metadata;
confidence score.
Rationale

These values are already known deterministically by the application.

Asking the LLM to reproduce them creates unnecessary hallucination risk.

Validation Required

Tests will verify that:

unknown citation IDs are rejected;
citations map only to retrieved chunks;
confidence is derived from retrieval/evidence signals rather than LLM self-reporting.

-----------------------------------------------------------------------
Decision 009 — Calibrate the no-answer threshold from evaluation data

Status: Planned — retrieval evaluation not yet implemented

Proposed Decision

Do not choose the retrieval relevance threshold as an arbitrary constant.

Estimate a practical threshold using both:

answerable Medicare questions;
deliberately out-of-document negative questions.
Rationale

The system should abstain when evidence is insufficient and should avoid invoking the LLM with irrelevant context.

Important Interpretation

The resulting confidence_score will be documented as an evidence-strength heuristic, not as a calibrated probability that the generated answer is correct.

Validation Required

The threshold will be finalized only after retrieval results for positive and negative evaluation queries are available.

---------------------------------------------------------------------
Decision 010 — Keep internal parser structures lightweight

Status: Accepted — implemented

Decision

Use Python dataclasses for internal PDF extraction structures and reserve Pydantic models for external validation boundaries.

Rationale

Objects such as extracted text lines and document units are created frequently during ingestion and are controlled entirely by internal code.

Pydantic is more valuable at boundaries such as:

API requests;
API responses;
LLM structured output.

Using Pydantic for every extracted PDF line would add validation overhead without meaningful benefit.

Implementation

app/models/document.py

-----------------------------------------------------------------
## Decision 011 — Reconstruct spacing using span geometry

**Status:** Accepted — implemented and regression-tested

### Decision

Do not concatenate adjacent PyMuPDF spans using an empty-string join.

Use explicit whitespace plus horizontal span geometry to determine whether a space belongs between adjacent spans.

### Evidence

Manual parser inspection exposed malformed text such as:

- `amountafter`;
- `ayearly`;
- `You canchoose`;
- `Important!Remember`;
- `Important!If`.

These occurred because visually separated words can be represented as separate PDF spans without whitespace characters.

### Trade-off

Blindly inserting spaces between every span could damage legitimate punctuation and constructs such as currency values or hyphenated terms.

The parser therefore combines explicit whitespace, geometric separation, and punctuation-aware rules.

### Implementation

`join_spans()` in:

`app/rag/pdf_parser.py`

### Validation

Regression tests verify the previously observed failures on physical pages 11, 17, and 80.

---------------------------------------------------------


Current Implementation Status
| Area                      | Status                                     |
| ------------------------- | ------------------------------------------ |
| Project/environment setup | Complete                                   |
| FastAPI health endpoint   | Complete                                   |
| PDF inspection            | Complete                                   |
| PDF structural parser     | Implemented; manual validation in progress |
| Dynamic chunking          | Not started                                |
| Chunk-size evaluation     | Not started                                |
| Embeddings                | Not started                                |
| FAISS retrieval           | Not started                                |
| Retrieval evaluation      | Not started                                |
| OpenRouter generation     | Not started                                |
| Citation validation       | Not started                                |
| Evidence confidence       | Not started                                |
| `/query` endpoint         | Not started                                |
| Docker                    | Not started                                |
| Deployment                | Not started                                |


This status table is updated as implementation progresses so that repository documentation does not claim functionality that has not been built or validated.
