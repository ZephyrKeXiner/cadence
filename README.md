# Cadence

Cadence is a local LLM inference experiment built with Swift, SwiftUI, and Metal Performance Shaders Graph (MPSGraph).

At its current stage, the repository is focused on building and validating the low-level pieces of a Transformer inference stack rather than shipping a finished chat application. The main pattern in the codebase is: implement a GPU operator with MPSGraph, then compare it against a CPU reference implementation.

## Status

Cadence is still in an early R&D phase. It already contains a set of core operators and some tokenizer groundwork, but it does not yet provide end-to-end model loading, decoding, or a usable chat experience.

Right now, the repository is best understood as:

- a Metal / MPSGraph operator playground
- a validation project that compares CPU and GPU outputs
- a skeleton for a future local inference engine

It is not yet:

- a production-ready chat app
- a complete Qwen inference runtime
- a mature project with a dedicated XCTest target

## What Is Implemented

### 1. Runtime and utilities

- `Cadence/Utils/Device.swift`
  - Centralized setup for `MTLDevice`, `MTLCommandQueue`, and `MPSGraphDevice`
- `Cadence/Utils/TensorUtils.swift`
  - Converts between `[Float]` and `MPSGraphTensorData`
  - Provides helpers such as `floatPlaceholder`, `floatConstant`, and `floatScalar`

### 2. Core Transformer operators

The following operators are already implemented:

- `Attention.apply`
  - Single-head scaled dot-product attention
- `Attention.applyMultiHead`
  - Multi-head attention
  - Supports causal masking
- `Attention.applyGQA`
  - Grouped Query Attention
  - Expands KV heads by group and reuses the multi-head attention path
- `RoPE.precomputeCosSin` / `RoPE.apply`
  - Rotary Position Embedding table precomputation and application
- `RMSNorm.apply`
  - RMSNorm implementation
  - Also exposes intermediate values `meanSquared` and `invRms`
- `LayerNorm.apply`
  - Standard LayerNorm
- `SWiGLU.apply`
  - SWiGLU feed-forward block commonly used in modern Transformers

### 3. Tokenizer groundwork

- `Cadence/Tokenizer/ByteShadowMap.swift`
  - Implements byte-level shadow character mapping
  - Encodes arbitrary UTF-8 byte sequences into a reversible character sequence
  - Supports both `encode` and `decode`

This is useful as a low-level building block before wiring in full BPE / tokenizer logic.

### 4. CPU vs GPU validation

The current validation approach is not XCTest-based. Instead, the test runners are compiled into the app target and manually called from `CadenceApp.init()`.

Available runners:

| Runner | Purpose |
| --- | --- |
| `MatmulTest.run()` | Verifies MPSGraph matrix multiplication against a CPU implementation |
| `RMSNormTest().run()` | Verifies RMSNorm |
| `RoPETest().run()` | Verifies RoPE numerically |
| `RoPEPropertyTest().run()` | Verifies that RoPE preserves vector length |
| `LayerNormTest.run()` | Verifies LayerNorm |
| `SWiGLUTest.run()` | Verifies SWiGLU |
| `AttentionTest.run()` | Verifies single-head attention |
| `AttentionTest.runMulti()` | Verifies multi-head attention |
| `AttentionTest.runGQA()` | Verifies GQA |
| `AttentionPerfTest.run()` | Rough CPU vs GPU attention performance comparison |
| `ByteShadowMapTest.run()` | Verifies byte-shadow mapping and round-trip encoding |

## Repository Layout

```text
.
├── Cadence
│   ├── CadenceApp.swift
│   ├── ContentView.swift
│   ├── Operator
│   │   ├── Attention.swift
│   │   ├── LayerNorm.swift
│   │   ├── RMSNorm.swift
│   │   ├── RoPE.swift
│   │   └── SWiGLU.swift
│   ├── Tokenizer
│   │   └── ByteShadowMap.swift
│   ├── Utils
│   │   ├── Device.swift
│   │   └── TensorUtils.swift
│   ├── Test
│   │   ├── OperatorTest
│   │   └── TokenizerTest
│   └── Assets.xcassets
├── Cadence.xcodeproj
├── Models
│   └── Qwen3.5-4B
└── LICENSE
```

## Requirements

Recommended environment:

- macOS
- a Mac with Metal support
- Xcode 26.1.1 or a compatible version
- Swift 5
- `swiftformat`

Notes:

- The project is currently configured with `MACOSX_DEPLOYMENT_TARGET = 26.1`
- The project includes a build phase that runs:
  - `/opt/homebrew/bin/swiftformat "$SRCROOT" --swiftversion 5.0`
- If `swiftformat` is not installed, install it with:

```bash
brew install swiftformat
```

If you do not want automatic formatting during builds, you can remove or disable that Run Script Build Phase in Xcode.

## Getting Started

### Open in Xcode

```bash
open Cadence.xcodeproj
```

Then:

1. Select the `Cadence` scheme
2. Select `My Mac` as the run destination
3. Run the app

### Build from the command line

If your machine does not have a `Mac Development` signing certificate configured, you can disable signing for local builds:

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

This command has already been verified successfully in this repository.

## How To Run The Existing Tests

The current entry point for manual test execution is `Cadence/CadenceApp.swift` inside `init()`.

At the moment, the default runner is:

```swift
ByteShadowMapTest.run()
```

To run a different test, edit `CadenceApp.init()`. For example:

```swift
init() {
    MatmulTest.run()
//    RMSNormTest().run()
//    RoPEPropertyTest().run()
//    LayerNormTest.run()
//    SWiGLUTest.run()
//    AttentionPerfTest.run()
//    AttentionTest.runGQA()
//    RoPETest().run()
//    ByteShadowMapTest.run()
}
```

The results will appear in:

- the Xcode debug console
- or the app's standard output logs at launch

## Model Assets

The repository includes a model asset directory:

```text
Models/Qwen3.5-4B/
```

The directory currently contains:

- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `model.safetensors-00002-of-00002.safetensors`

Important caveats:

- These assets are not yet loaded or used by the Swift code
- The repository currently only contains the `00002-of-00002` safetensors shard
- Based on the filename, the full model weights are likely missing the first shard

So while the repository is clearly preparing for Qwen3.5-4B support, the full path for:

- weight loading
- tokenizer parsing
- embedding / block assembly
- KV cache
- sampling / generation

has not been wired up yet.

## Current Limitations

As of now, Cadence still has several obvious gaps:

- `ContentView.swift` is still the default template UI with `Hello, world!`
- there is no complete safetensors reader
- tokenizer files are not yet parsed into a usable tokenizer
- there is no end-to-end Transformer block forward pass
- there is no logits sampling or text generation
- the tests have not been moved into a dedicated XCTest target

Because of that, the repository currently works best as:

- a low-level operator validation project
- a local inference engine prototype
- a Metal / MPSGraph learning and experimentation codebase

## Suggested Next Steps

If the goal is to turn this into a model that can run end to end, the most natural next steps are:

1. Add safetensors weight loading
2. Add tokenizer vocab and BPE merge parsing
3. Compose Attention, RoPE, RMSNorm, and SWiGLU into a full Transformer block
4. Add embeddings, LM head, and KV cache
5. Wire prompt -> token IDs -> logits -> sampling -> text output
6. Move the handwritten runners into XCTest and a proper benchmark setup

## Code Style

The repository currently uses `swiftformat`, configured in:

```text
.swiftformat
```

Current settings include:

- 4-space indentation
- maximum line width of 110
- consistent argument and parameter wrapping
- Swift version 5.0

## License

This project is licensed under the [MIT License](LICENSE).
