# MilCOD7K — Dataset Datasheet

This datasheet documents the construction, sources, annotation, and ethics of the MilCOD7K dataset. It accompanies the paper *Multi-Class Camouflaged Military Object Detection via Spatial–Frequency Collaborative Network with Category Prototype Contrast* and provides the dataset-transparency details requested in the review.

## 1. Overview

MilCOD7K is a benchmark for multi-class camouflaged military object segmentation. It contains 7,200 pixel-level annotated images divided into 5,760 training, 720 validation, and 720 test images. The label set comprises a background class and four foreground classes: camouflaged soldier, military vehicle, tank, and defense fortification. The dataset mixes real photographs with AI-generated images: 4,944 real and 2,256 AI-generated. A per-image real/AI label and the split assignment of every image are provided in `metadata.csv`.

## 2. Data sources

Real photographs are sampled from existing public military and camouflage datasets, namely MHCD2022, ACD1K, MTSD1K, and KIIT-MiTA, and are supplemented with images returned by public web image search services such as Bing and Baidu and with frames extracted from publicly available videos on platforms such as Bilibili and Haokan Video. AI-generated images are produced with the CamDiff diffusion pipeline, which is built on the RunwayML Stable Diffusion Inpainting latent diffusion model, and with public AI image-generation services including Doubao and Nano-banana-1. All sources are public and, to the authors' knowledge, carry no copyright restriction that prevents academic research use.

## 3. Real and AI composition

The dataset contains 4,944 real photographs and 2,256 AI-generated images. In the training set 1,774 of 5,760 images are AI-generated, in the validation set 300 of 720, and in the test set 182 of 720. The per-image real/AI label of every image is recorded in `metadata.csv`, so the split is fully reproducible. The AI proportion is uneven across classes: the soldier class contains no AI-generated images, while defense fortification has the highest AI share.

## 4. Synthetic-image generation

The AI-generated images come from two routes. The diffusion route uses CamDiff, a camouflage-augmentation pipeline built on the RunwayML Stable Diffusion Inpainting latent diffusion model, with prompt templates centered on realistic combat scenes, such as complex battlefield environment, camouflaged tank, military vehicle, defense fortification, severe weather, and night. The remaining AI images are produced with the public services Doubao and Nano-banana-1 using similar prompts. Candidates from both routes are filtered to remove near-duplicates and images that are clearly inconsistent with a realistic battlefield; the CamDiff pipeline additionally applies CLIP zero-shot classification to discard failed syntheses whose content does not match the prompt. We note that the outputs of commercial AI services such as Doubao and Nano-banana-1 are not version-pinned and are therefore not exactly reproducible. Distribution consistency between the synthetic and real subsets is verified on the training set with five complementary metrics reported in the paper: FID 103.54, CLIP cosine similarity 0.83, Inception Score 8.23 for real and 6.39 for AI images, RGB color-histogram correlation 0.87/0.83/0.81, and Wasserstein distance 0.53/0.53/0.51. These values indicate acceptable distribution similarity, strong semantic alignment, and comparable visual quality between the two subsets.

## 5. Annotation

Pixel-level masks were produced by a single annotator with military-domain experience using the labelme polygon tool, following a written guideline that fixes the class boundaries described below. Masks are stored as one-hot arrays of shape H × W × 5. Annotation is single-expert, so inter-annotator agreement is not reported; objects that could not be unambiguously assigned were discarded. Single-annotator labeling is acknowledged as a limitation, and multi-annotator labeling with reported agreement, for example Cohen's κ or mean IoU, is recommended for future releases.

## 6. Class definitions

The four foreground classes are camouflaged soldier, military vehicle, tank, and defense fortification. The boundary between military vehicle and tank is defined by equipment structure: a tank possesses tracks and a gun barrel, which are structurally distinct from mostly wheeled military vehicles. Defense fortification covers bunkers, camouflage-net-covered emplacements, typical defensive works, and camouflage barriers. Pixel-level rather than bounding-box annotation is used throughout, in order to support semantic segmentation.

## 7. Duplicate removal and split hygiene

All 7,200 images were checked for near-duplicates using perceptual hashing with a 256-bit hash combined with structural similarity at a threshold of 0.90. No near-duplicate pairs were found, with zero duplicates within a split and zero leakage across splits, so the train, validation, and test splits are disjoint. The same procedure was applied between MilCOD7K and the MHCD2022 test set, where one overlapping image was found and excluded.

## 8. License and ethics

The dataset is released under CC-BY-NC 4.0 for non-commercial research use, and the accompanying code is released under MIT. The dataset targets defensive applications such as reconnaissance, situational awareness, and analyst training support. It contains no real unit identifiers, deployment locations, or classified equipment, and the synthetic images depict only generic prompted scenes with no relation to any specific real weapon system. Construction and release comply with the research-ethics requirements of the authors' institution and the funding agency.

## 9. Known limitations

Annotation is single-annotator and inter-annotator agreement is not reported. Precise per-channel source counts were not tracked individually. The AI proportion is uneven across classes. Foreground occupies only 11.55% of all pixels, reflecting the inherent class imbalance of camouflage scenes.
