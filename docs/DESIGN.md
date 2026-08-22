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
## Decision 007 — Derive and Empirically Select Chunk Size from Document Statistics

**Status:** Accepted — document-derived candidate generation, structure-aware
chunk construction, retrieval evaluation, and final strategy selection completed

### Decision

Do not use a user-defined or arbitrarily hardcoded chunk size.

The system automatically derives multiple candidate chunk targets from measured
characteristics of the supplied document. Each candidate is constructed using
the same structure-aware chunking algorithm and evaluated against an
independently labeled retrieval benchmark.

The final chunking strategy is selected empirically during the indexing stage
using retrieval-quality metrics rather than being predefined by the user.

Query-level chunk-size adaptation is not required.

This design follows the assignment clarification that dynamic chunking may be
optimized during indexing by considering semantic boundaries, content
structure, multiple chunk sizes, and retrieval metrics.

### Evidence from Document Profiling

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

These measurements show that normal semantic units can generally remain intact,
while unusually large structural sections must be divided at semantic-unit
boundaries.

### Structural Section Reconstruction

The chunking pipeline preserves document structure rather than dividing text
solely by character or token count.

Consecutive heading lines on the same physical page are grouped so that
visually split headings remain one logical heading.

A heading appearing on a new physical page begins a new structural section when
the preceding section contains only heading material. This prevents unrelated
page-level headings from being merged merely because no paragraph occurs
between them.

Paragraphs and list items remain atomic semantic units whenever they fit within
the embedding model's hard token constraint.

This behavior was refined through manual inspection of the supplied Medicare
document.

### Candidate Derivation

Candidate targets are generated from measured document statistics rather than
being supplied by the user or stored as a fixed configuration list.

The derivation considers:

- the greater of structural-section median and paragraph P95;
- structural-section P75;
- structural-section P90;
- structural-section P95;
- the embedding model's maximum token length.

The resulting raw values are:

1. normalized to the nearest practical 32-token boundary;
2. deduplicated;
3. sorted;
4. capped by the embedding model's token constraint.

For the supplied Medicare document, this algorithm produced:

`128, 192, 320, 416`

These values are outputs of the derivation logic. They are not hardcoded
chunk-size choices.

### Candidate Corpus Construction

Every candidate target is processed using the same structure-aware chunking
algorithm. This ensures that retrieval evaluation primarily compares chunk-size
behavior rather than different chunking implementations.

The target token count is treated as a soft packing objective.

Complete paragraphs and list items are preserved whenever they fit within the
embedding model's hard limit. Large structural sections are divided by greedily
packing complete semantic units.

Leading heading context is retained for continuation chunks so that chunks do
not lose their section meaning when embedded independently.

The embedding model's 512-token maximum is treated as a hard constraint.

Candidate construction produced:

| Target | Chunks | Mean tokens | Median tokens | Max tokens | Above soft target |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 798 | 86.82 | 92 | 260 | 53 |
| 192 | 632 | 107.11 | 109 | 260 | 14 |
| 320 | 523 | 127.16 | 106 | 320 | 0 |
| 416 | 481 | 137.37 | 106 | 411 | 0 |

Chunks exceeding the 128- or 192-token soft target occur when preserving a
complete semantic unit is preferable to destructively splitting it merely to
satisfy the target.

No generated chunk exceeds the 512-token hard limit.

### Retrieval Evaluation Design

The candidate strategies were evaluated against a manually verified retrieval
benchmark rather than against labels inferred from the retriever itself.

Twelve positive evaluation queries were created across multiple Medicare
information categories, including:

- eligibility;
- enrollment;
- coverage;
- calculation;
- appeals;
- plan rules;
- penalty avoidance;
- multi-evidence questions.

Gold evidence was labeled against the original pre-chunk semantic source-unit
identifiers.

This is important because chunk identifiers differ between candidate
strategies. Using the original semantic source units as ground truth allows the
128-, 192-, 320-, and 416-token corpora to be compared against the same
underlying evidence.

Multi-part answers are represented using evidence groups so that retrieval can
be evaluated according to how much of the required answer evidence is actually
recovered.

For example, the Medicare Advantage eligibility query contains independent
evidence groups for:

- having Part A and Part B;
- living in the plan's service area;
- being a U.S. citizen or lawfully present in the U.S.

This prevents retrieval of only one part of a multi-part answer from being
treated as complete evidence recovery.

### Retrieval Infrastructure

Each candidate corpus was embedded using:

`BAAI/bge-small-en-v1.5`

Document embeddings are normalized before indexing.

Queries use the BGE retrieval instruction before embedding.

Each candidate corpus is stored in an exact FAISS:

`IndexFlatIP`

index.

Because the embeddings are L2-normalized, inner-product ranking behaves as
cosine-similarity ranking.

Exact search was chosen instead of approximate nearest-neighbor indexing
because each candidate corpus contains fewer than 1,000 chunks. At this scale,
exact retrieval is simple, deterministic, and avoids unnecessary ANN
hyperparameter tuning.

### Evaluation Metrics

The following retrieval metrics were calculated:

- Precision@1;
- Precision@3;
- Precision@5;
- Recall@1;
- Recall@3;
- Recall@5;
- evidence-group Recall@5;
- Mean Reciprocal Rank (MRR@5);
- NDCG@5.

Recall@K measures recovery of manually identified source evidence.

Evidence-group Recall@5 additionally measures coverage of distinct logical
answer components.

MRR@5 measures how early the first relevant evidence appears.

NDCG@5 measures ranking quality within the retrieved results.

Precision metrics provide an additional indication of retrieval noise.

### Empirical Results

The four automatically derived chunk strategies produced the following
aggregate retrieval results:

| Strategy | P@1 | P@5 | R@1 | R@3 | R@5 | Group R@5 | MRR@5 | NDCG@5 | Mean chunk tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target_128 | 0.4167 | 0.2167 | 0.3403 | 0.5625 | 0.6806 | 0.6972 | 0.5444 | 0.5725 | 86.8 |
| target_192 | 0.5000 | 0.2500 | 0.4236 | 0.6875 | 0.8532 | 0.8444 | 0.6667 | 0.7000 | 107.1 |
| target_320 | 0.3333 | 0.2167 | 0.2847 | 0.7698 | 0.8532 | 0.8444 | 0.5903 | 0.6322 | 127.2 |
| **target_416** | **0.5833** | 0.2000 | **0.5417** | **0.7976** | **0.8810** | **0.8611** | **0.7153** | **0.7372** | 137.4 |

The 416-token strategy achieved the strongest result on the primary retrieval
quality measures:

- highest Precision@1: `0.5833`;
- highest Recall@1: `0.5417`;
- highest Recall@3: `0.7976`;
- highest Recall@5: `0.8810`;
- highest evidence-group Recall@5: `0.8611`;
- highest MRR@5: `0.7153`;
- highest NDCG@5: `0.7372`.

Although its Precision@5 is lower than the 192-token strategy, this is expected
when larger chunks consolidate several relevant source units into a smaller
number of evidence-bearing chunks.

For RAG generation, evidence recovery and early placement of relevant evidence
are more important than maximizing the number of individually relevant chunks
among all five retrieved results.

### Strategy-Selection Policy

Selection is deterministic and uses the following priority order:

1. highest mean Recall@5;
2. highest mean MRR@5;
3. highest mean NDCG@5;
4. highest mean Precision@5;
5. lower mean chunk-token count as the final tie-breaker.

The selection policy therefore prioritizes evidence recovery first, followed by
ranking quality, before considering retrieval precision and context cost.

Under this policy, the empirically selected strategy is:

`target_416`

with a soft target of:

`416 tokens`

The selected corpus contains:

- 481 chunks;
- mean chunk length of approximately 137.37 tokens;
- maximum generated chunk length of 411 tokens.

### Production Strategy

Production code does not hardcode `target_416`.

After evaluation, the selected candidate index is copied to the stable
production alias:

`artifacts/indexes/selected/`

The application can therefore load the empirically selected index without
knowing which candidate target won the evaluation.

If the document, embedding model, chunking algorithm, or evaluation dataset
changes and a different candidate wins, the production retrieval code does not
need to change.

The selected strategy and supporting metrics are persisted in:

`artifacts/selected_strategy.json`

### Rationale

Dynamic chunk-size selection is a primary requirement of the assignment.

Using a conventional fixed value such as 256 or 512 tokens and describing it as
dynamic would not demonstrate document-adaptive behavior.

The implemented approach separates three concerns:

1. candidate derivation from document statistics;
2. structure-aware construction of comparable candidate corpora;
3. empirical selection based on retrieval quality.

This makes chunk-size selection:

- document-adaptive;
- reproducible;
- measurable;
- explainable;
- independent of user configuration.

The results also demonstrate that candidate selection matters empirically.

For example, Recall@5 increased from approximately `0.681` for the 128-token
strategy to approximately `0.881` for the selected 416-token strategy.

The final chunk size is therefore not based on intuition or an arbitrary
default. It is the result of a repeatable indexing-stage optimization process.

### Validation Completed

The implementation verifies that:

- consecutive heading lines on the same page remain together;
- headings crossing into a new physical page are separated when appropriate;
- candidate targets are derived from supplied document statistics;
- candidate values are normalized to practical token boundaries;
- duplicate targets are removed;
- candidate targets respect embedding-model constraints;
- semantic units are preserved rather than unnecessarily split;
- continuation chunks retain heading context;
- chunk identifiers are deterministic and unique;
- source-unit coverage is preserved;
- page provenance is retained;
- candidate chunks are non-empty;
- no generated chunk exceeds the 512-token hard limit;
- document embeddings are normalized;
- FAISS metadata remains aligned with vector rows;
- persisted indexes can be reloaded and queried;
- retrieval queries are embedded independently of document chunks;
- gold evidence refers to pre-chunk semantic source units;
- all manually labeled gold units exist in the parsed document;
- labeled evidence pages match the original PDF;
- all candidate indexes are evaluated using identical queries;
- retrieval metrics are calculated deterministically;
- final strategy selection follows an explicit metric hierarchy;
- the selected production index can be queried successfully.

A production smoke test against the selected index retrieved the Medicare Part D
late-enrollment penalty evidence from pages 83–84 at rank 1.

### Implementation

Implemented chunking and profiling:

- `app/rag/chunking.py`
- `app/rag/tokenization.py`
- `scripts/profile_token_lengths.py`
- `scripts/build_candidate_chunks.py`
- `tests/test_tokenization.py`
- `tests/test_chunking.py`

Implemented embedding and retrieval:

- `app/rag/embeddings.py`
- `app/rag/vector_store.py`
- `app/rag/retrieval.py`
- `scripts/build_candidate_indexes.py`
- `tests/test_embeddings.py`
- `tests/test_vector_store.py`
- `tests/test_retrieval.py`

Implemented evaluation and strategy selection:

- `app/models/evaluation.py`
- `app/rag/evaluation.py`
- `evaluation/golden_queries.json`
- `scripts/validate_golden_queries.py`
- `scripts/evaluate_chunk_strategies.py`
- `scripts/select_chunk_strategy.py`
- `tests/test_evaluation.py`

Generated reproducibility/evaluation evidence:

- `artifacts/token_profile.json`
- `artifacts/chunk_strategy_profile.json`
- `artifacts/index_profile.json`
- `artifacts/retrieval_evaluation.json`
- `artifacts/selected_strategy.json`

Generated candidate chunk corpora under:

`artifacts/chunks/`

and generated FAISS indexes under:

`artifacts/indexes/`

are reproducible build artifacts and are not committed to source control.

### Outcome

Dynamic chunking is considered complete.

The indexing pipeline automatically derives candidate chunk targets from
document statistics, preserves semantic structure while constructing each
candidate corpus, evaluates all candidates with independently labeled retrieval
queries, and selects the strongest candidate according to predefined retrieval
metrics.

For the supplied Medicare document and
`BAAI/bge-small-en-v1.5`, the selected strategy is:

**416-token target with structure-aware semantic chunking.**

This decision should only be revisited if a later change to the source
document, embedding model, chunk-construction algorithm, or retrieval benchmark
produces materially different evaluation results.
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
