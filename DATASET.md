# Text pretraining corpus

The production byte model is planned for **100B GPT-2-token equivalents**.
That is a token-equivalent accounting target only; the model trains directly
on UTF-8 bytes. The manifest must record the realized raw-byte total as well.
For the proposed English-heavy mixture, 100B token equivalents will usually
require roughly 300–500B UTF-8 bytes, or about 73–122M 4096-byte windows.

The initial mixture is:

| component | sampling share | pinned dataset revision | license metadata |
| --- | ---: | --- | --- |
| recent FineWeb dumps | 45% | `HuggingFaceFW/fineweb@9bb295ddab0e05d785b879661af7260fed5140fc` | ODC-By 1.0 / Common Crawl terms |
| FineWeb-Edu score ≥3 | 20% | `HuggingFaceFW/fineweb-edu@87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By 1.0 / Common Crawl terms |
| score-2-only educational text | 10% | same FineWeb-Edu revision | ODC-By 1.0 / Common Crawl terms |
| curated Dolma non-web text | 10% | `allenai/dolma@7f48140530a023e9ea4c5cfb141160922727d4d3` | ODC-By plus source terms |
| FineWeb2 multilingual text | 10% | `HuggingFaceFW/fineweb-2@af9c13333eb981300149d5ca60a8e9d659b276b9` | ODC-By 1.0 / Common Crawl terms |
| other clearly licensed reference text | 5% | pin each source separately | source-specific |

FineWeb-Edu is derived from FineWeb. Assign a document to the strict-Edu tier
first, then the score-2-only tier, and only then to the base FineWeb pool. Run
exact and near-duplicate removal across all tiers before sampling and before
the train/validation split. Split by document or repository, not by adjacent
byte ranges. Keep the validation sources and benchmark decontamination list
outside the training payload.

The trainer consumes a manifest-backed payload. The manifest is a JSON object
with `version: 1`, a relative `data_path`, `data_size`, `data_sha256`, and
`total_token_equivalents`, followed by `documents`. Every document entry has
`id`, `offset`, `length`, `sha256`, `token_count`, `source`, `license`, and
`language`; quality scores, dump/date, URL, and dedup cluster identifiers should
also be retained. Offsets must refer to non-overlapping byte ranges in the
payload. The corpus builder must compute each document hash after filtering,
validate UTF-8, and write the payload and manifest atomically.

The loader samples one document-local window per batch row and right-pads short
documents. This keeps CUDA graph shapes fixed and prevents attention, H-Net
routing, FBT, MSAR, or recurrent KV state from crossing document boundaries.
Do not concatenate unrelated documents into a row with a textual separator.

The repository contains code and the manifest schema only. Downloaded shards,
payloads, manifests for a particular run, and checkpoints belong outside the
repository. A run records the payload and manifest hashes in its provenance;
resuming with a changed corpus is rejected.
