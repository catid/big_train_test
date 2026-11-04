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

For the full 100B-token build, use
`tools/build_byte_corpus.py --target-token-equivalents 100000000000
--compact-index`. Token-equivalent budgets use each record's pinned
`token_count` field and preserve the source mixture quotas. Use
`--target-bytes` when the desired budget is raw UTF-8 payload bytes instead;
the two budget options are mutually exclusive. A token budget is preferred
for the production 100B target because the trainer reports progress in the
same accounting unit while still recording realized byte totals in each
manifest.

This writes a version-2 manifest plus a fixed-width `HNETIDX2` sidecar containing
byte offset, byte length, token-equivalent count, and document SHA-256 for each
row. The trainer memory-maps that sidecar and draws a deterministic virtual
payload byte for each global batch row, mapping it with `searchsorted` to a
document range. This weights sampling by document byte length without an O(N)
cumulative table or a full `randperm` tensor. Version-1 JSON manifests remain
valid for small pilots and compatibility checks; a non-contiguous payload uses a
bounded cumulative window table instead.

Each manifest records the source-spec SHA-256, sampling seed, validation
fraction, deduplication mode, and builder commit. Validation grouping is
source-independent: repository identifiers or normalized URL hosts (including
`www.` variants) are assigned together so derived FineWeb-Edu records cannot
place the same host in opposite splits.

Run `tools/benchmark_byte_stream.py` before a long job to verify the actual
manifest and batch shape. The validated eight-GPU B300 path at `B=18, S=4096`
writes directly into persistent CUDA-graph inputs and measured `0.56 ms` per
batch at the slowest rank over 200 warmed steps in the latest run (the exact
value depends on page-cache state and host contention). The stream must keep
active and loss masks enabled
for the production mixture: the loader's document-local pilot had only 23.1%
of documents long enough to fill a complete 4096-byte window.

The trainer's distributed default uses `--manifest-hash-mode rank0`: rank zero
hashes the large payload, compact index, and metadata sidecar once, while every
rank still performs local size, path, header, range, and token-total checks and
compares the declared identity before entering model construction. Use
`--manifest-hash-mode strict` when ranks do not share an immutable filesystem;
that mode hashes each file on every rank.

The active 100B build uses exact and normalized deduplication across all three
available pools. The conservative SimHash near-dedup pass remains available for
smaller audits; its SQLite locality-sensitive index is too slow for a
multi-hundred-gigabyte production pass.
