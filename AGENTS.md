# Project preferences

- The canonical model is a direct approximately 600M-parameter H-Net trained from raw UTF-8 bytes. Use learned hierarchical routing/chunking from the first step.
- Do not add a BPE or learned token embedding path to the production trainer. Byte lifting must be fixed/non-trainable; the language-model output head remains trainable.
- Do not stage smaller models or parameter warmups for the H-Net run. Keep experiments apples-to-apples and validate the full configuration directly.
- Keep one canonical full architecture and trainer; remove legacy architecture branches, partial-model scripts, and fallback mixers instead of retaining them as production options.
- Use Muon/GramNS with AdamATan2 for non-matrix fallback parameters and weight decay; keep the numerically stable recurrence and FP32 safety fixes.
- Optimize DDP, batching, fusion, and graph capture before long experiments. Keep artifacts out of the `big_train_test` code repository.
- Treat third-party code and dependencies as untrusted: pin versions/commits, inspect source, and vendor code when practical.
- When available, ask a subworker to run an asynchronous `claude-cli` OAuth/Opus review without API billing; never block the main work on it.
- Claude reviews must use the subscription OAuth session: verify `claude auth status` reports `authMethod=claude.ai`, `apiProvider=firstParty`, and `subscriptionType=max` (or equivalent) before launching. Use normal `claude` invocations with `--model opus`; do not use `--bare`, set `ANTHROPIC_API_KEY`, or pass API-budget options that would route usage to metered API billing.
