# Cadence

Cadence is an experimental local LLM runtime for macOS, built in Swift on top of Metal Performance Shaders Graph (MPSGraph).

The project is focused on building the core pieces of a Transformer stack as small, testable components: weight loading, tokenization, and GPU operators. The current workflow is intentionally low-level: implement a component, validate it in isolation, then compose upward.

## Highlights

- Transformer operators in MPSGraph
  - scaled dot-product attention
  - multi-head attention
  - grouped-query attention (GQA)
  - RoPE
  - RMSNorm and LayerNorm
  - SWiGLU
- Tokenization pipeline
  - byte-level reversible shadow mapping
  - vocab and merges loading
  - GPT/Qwen-style BPE
  - special token handling
  - encode / decode round-trip support
- SafeTensors support
  - header parsing
  - `F32` and `BF16` decoding to `Float32`
  - shard routing through `model.safetensors.index.json`
- Validation runners
  - CPU vs GPU operator checks
  - tokenizer correctness checks
  - real-model safetensors smoke tests

## Status

Cadence is still an early prototype. It can validate important parts of a Qwen-style inference stack, but it does not yet run full end-to-end generation.

Current gaps:

- no full Transformer forward pass
- no KV cache
- no logits sampling or decoding loop
- no production UI
- no dedicated XCTest target yet

## Repository Layout

```text
.
├── Cadence
│   ├── Loader        # safetensors parsing and shard routing
│   ├── Operator      # MPSGraph operators
│   ├── Tokenizer     # byte mapping and BPE vocab logic
│   ├── Test          # manual validation runners
│   ├── Utils         # device and tensor helpers
│   ├── CadenceApp.swift
│   └── ContentView.swift
├── Cadence.xcodeproj
├── .swiftformat
└── LICENSE
```

## Requirements

- macOS
- a Metal-capable Mac
- Xcode 26.1.1 or a compatible version
- Swift 5
- `swiftformat`

The project currently runs `swiftformat` from an Xcode build phase:

```bash
brew install swiftformat
```

## Build

Open the project in Xcode:

```bash
open Cadence.xcodeproj
```

Or build from the command line:

```bash
xcodebuild \
  -project Cadence.xcodeproj \
  -scheme Cadence \
  -configuration Debug \
  -derivedDataPath /tmp/CadenceDerivedData \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  build
```

If you build locally without a configured `Mac Development` signing certificate, keep the signing flags above.

## Local Model Assets

Model assets are intentionally not tracked in git. The `Models/` directory is ignored.

To run the tokenizer and safetensors tests against a real model, place your local files under a model directory and update [Cadence/Test/TestPaths.swift](/Users/sakruhnab1/Documents/Cadence/Cadence/Test/TestPaths.swift).

The current test helpers expect files such as:

- `vocab.json`
- `merges.txt`
- `model.safetensors.index.json`
- one or more `.safetensors` shards

## Running Validation Runners

Manual validation is wired through [Cadence/CadenceApp.swift](/Users/sakruhnab1/Documents/Cadence/Cadence/CadenceApp.swift). The current default runner is:

```swift
SafetensorsRouterTest.run()
```

To run a different check, change the runner in `CadenceApp.init()`.

Common runners:

- Loader
  - `SafeTensorsTest`
  - `SafeTensorsRealTest`
  - `SafetensorsRouterTest`
- Tokenizer
  - `ByteShadowMapTest`
  - `BPETest`
  - `SpecialTokenTest`
  - `TokenizerTest`
- Operators
  - `MatmulTest`
  - `RMSNormTest`
  - `RoPETest`
  - `RoPEPropertyTest`
  - `LayerNormTest`
  - `SWiGLUTest`
  - `AttentionTest`
  - `AttentionPerfTest`

## Roadmap

- compose the existing operators into full Transformer blocks
- remove hardcoded local model paths from test configuration
- add embeddings, KV cache, and a decoding loop
- move runner-based checks into XCTest and benchmarks
- replace the placeholder UI with a real inference interface

## License

This project is licensed under the [MIT License](LICENSE).
