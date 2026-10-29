# Project preferences

- The canonical model is a direct approximately 600M-parameter H-Net trained from raw UTF-8 bytes. Use learned hierarchical routing/chunking from the first step.
- Do not add a BPE or learned token embedding path to the production trainer. Byte lifting must be fixed/non-trainable; the language-model output head remains trainable.
- Do not stage smaller models or parameter warmups for the H-Net run. Keep experiments apples-to-apples and validate the full configuration directly.
- Every production training run must instantiate and train the complete selected architecture: fixed byte lifting, hierarchical routing/chunking, MSAR placement, recurrent inner loops with their cache policy, FBT objective, sparse MoE/shared expert, CroissAttn/FP8 path, and the selected optimizer. Do not substitute a GPT control, embedding model, dense FFN, or other partial architecture and report it as the full-system run.
- Architecture choices belong in the model configuration and must be exercised together in an end-to-end full-size training run. Unit tests, tiny smoke models, and isolated component checks are validation only; they do not replace the full-system run.
- Before claiming a training objective is complete, record the exact model/configuration used by the run and verify that its checkpoint was produced by that complete configuration.
- Use Muon/GramNS with AdamATan2 for non-matrix fallback parameters and weight decay; keep the numerically stable recurrence and FP32 safety fixes.
- Optimize DDP, batching, fusion, and graph capture before long experiments. Keep artifacts out of the `big_train_test` code repository.
- Treat third-party code and dependencies as untrusted: pin versions/commits, inspect source, and vendor code when practical.
- When available, ask a subworker to run an asynchronous `claude-cli` OAuth/Opus review without API billing; never block the main work on it.
