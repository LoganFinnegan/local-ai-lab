# Experiments

## SDXL Model Comparison

### Goal

Compare two local image-generation models while keeping the hardware,
prompt, resolution, and random seed consistent.

### Models

- SDXL Lightning 4-step
- Juggernaut XL v9

### Environment

- Radeon RX 9070 XT, 16 GB VRAM
- Ubuntu Linux
- ROCm
- PyTorch
- ComfyUI
- 1024 × 1024 images

### Configuration

#### SDXL Lightning
- Steps: 4
- CFG: 1
- Sampler: Euler
- Scheduler: sgm_uniform

#### Juggernaut XL
- Steps: 30
- CFG: 5
- Sampler: DPM++ 2M
- Scheduler: Karras

### Observations

Lightning generated images faster and tended toward a stylized or
illustrated appearance.

Juggernaut produced substantially more photorealistic images, but could
produce severe anatomy and object-detail errors.

Both models could identify broad concepts in prompts but struggled with
precise relationships between multiple objects.

Examples included:
- A cat and mouse prompt produced both subjects, but often placed the
  mouse incorrectly.
- A fisherman, net, and alligator prompt frequently omitted the
  fisherman and net or generated extra alligators.
- Complex structures such as rope and knots were unreliable.

### Takeaway

Image quality and prompt adherence are separate problems. A model can
produce a visually convincing image without accurately representing all
requested objects or their relationships.

Model selection and model-specific inference settings had a major effect
on output even though the underlying hardware and software stack stayed
the same.
