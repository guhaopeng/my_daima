# Part-aware VAE Plan

## Goal

Train a hierarchical Part-aware VAE on top of a frozen `PartSDF` decoder so that the optimized latent space is:

- more continuous than the original auto-decoder part latents
- more stable during physics optimization
- less likely to produce disconnected or collapsed intermediate shapes

## Available Inputs

The current `data/chair` layout already provides the core supervision needed for a first version:

- global surface points: `data/chair/samples/<instance>/surface.npy`
- full-shape SDF samples: `data/chair/samples/<instance>/deepsdf.npz`
- part meshes: `data/chair/parts/meshes/<instance>/part*.obj`
- part pose parameters: `data/chair/parts/parameters/<instance>/quaternions.npy`, `translations.npy`, `scales.npy`
- pre-trained `PartSDF` checkpoint:
  - `experiments/checkpoint_chair/chair/model/model_2000.pth`
  - `experiments/checkpoint_chair/chair/latent/latents_2000.pth`
  - `experiments/checkpoint_chair/chair/latent/poses.pth`

## Phase 1

Implement a first trainable baseline with the following design:

1. Freeze the pre-trained `PartSDF` decoder.
2. Train a new `Part-aware VAE` that:
   - encodes global surface points
   - uses part pose tokens to preserve part relations
   - predicts per-part latent codes and per-part poses
3. Feed the predicted part latents and poses into the frozen `PartSDF`.
4. Supervise both:
   - latent/pose reconstruction
   - final SDF reconstruction

## First Version Architecture

### Encoder

- global point encoder on `surface.npy`
- part token builder from:
  - global feature
  - part pose
  - learned part id embedding
- lightweight transformer over part tokens

### Latent Variables

- global latent `z_g`
- per-part latent `z_p^i`
- no explicit relation latent in the first version

### Decoder

- decode `z_g` and all `z_p^i` into:
  - predicted part latents
  - predicted part poses
- pass them to frozen `PartSDF` to get final SDF values

## Losses

The first training version should include:

1. `L_sdf`: final SDF reconstruction loss on sampled query points
2. `L_part_lat`: regression loss on target part latents from the pre-trained checkpoint
3. `L_pose`: regression loss on target part poses
4. `L_kl`: KL regularization for the VAE latent distributions

Optional future additions:

- part relation consistency loss
- contact/adjacency regularization
- geometry continuity priors around structural joints

## Data Pipeline Tasks

1. Build a dataset that loads:
   - instance id
   - global surface points
   - SDF supervision points
   - raw part pose parameters
   - target part latents
   - target checkpoint poses
2. Build an `instance_id -> latent row` mapping using the full training split order that was used by the `PartSDF` checkpoint.
3. Support train/valid subsets such as:
   - `train_train.json`
   - `train_valid.json`

## Code Tasks

1. Add a new model file:
   - `src/model/part_vae.py`
2. Register the new model in:
   - `src/model/__init__.py`
3. Add a new dataset class in:
   - `src/data.py`
4. Add a training script:
   - `scripts/train_part_vae.py`
5. Add a config template:
   - `experiments/template/specs_part_aware_vae.json`

## Validation Goals

After the first version trains, validate:

1. latent/pose reconstruction loss decreases stably
2. decoded shapes remain chair-like during interpolation
3. optimization in the new latent space is more stable than:
   - original `DeepSDF`
   - original `PartSDF` auto-decoder latent
   - single-global-latent VAE baseline

## Important Caveat

Even if this model trains well, it does **not** guarantee that the chair back will never partially disappear.

Reasons:

- if the load case does not require the back structurally, the optimizer may still reduce it
- the VAE prior only improves the search space, it does not enforce full semantic preservation
- mesh continuity also depends on decoder behavior, SDF thresholding, and physics constraints

Expected improvement:

- fewer catastrophic collapses
- smoother intermediate shapes
- less severe disconnected mesh artifacts

Not guaranteed automatically:

- zero disappearance
- perfect semantic preservation of all parts

## Next Steps After Phase 1

1. Add explicit relation latent `z_r`
2. Add part surface encoders from `parts/meshes`
3. Add relation regularization between seat, back, and legs
4. Connect the trained Part-aware VAE to `physiopt` and optimize only the VAE latent first
