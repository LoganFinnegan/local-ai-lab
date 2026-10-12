# Local AI Lab

A hands-on project exploring local AI inference, backend integration, and DevOps concepts on consumer AMD hardware.

## Goal

Build and understand a local AI stack that can run generative models without relying on an external AI API, then expose that capability through application code and backend services.

The project emphasizes:
- local/private inference
- AMD GPU acceleration with ROCm
- model experimentation and benchmarking
- backend/API integration
- reproducible setup and troubleshooting
- containerization, testing, and CI as the project develops

## Current Hardware

- AMD Radeon RX 9070 XT — 16 GB VRAM
- AMD Ryzen 7 9800
- 32 GB RAM
- Ubuntu Linux

## Current Stack

Ubuntu → AMDGPU → ROCm → PyTorch → ComfyUI → AI model

## Current Progress

### Image Generation
- Configured local GPU inference on AMD/ROCm
- Ran SDXL Lightning locally
- Added Juggernaut XL v9
- Built a side-by-side ComfyUI workflow to compare both models
- Tested identical prompts, resolution, and seeds
- Observed differences in speed, realism, prompt adherence, and failure modes

### Next
- Run a local LLM
- Access the LLM programmatically
- Build an API/backend around local inference
- Add private-document retrieval
- Containerize and automate testing

## Status

Work in progress.
