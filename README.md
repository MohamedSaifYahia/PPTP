# Adversarial Attacks on YOLOv5+: A Migration and Evaluation Project

## Repositories of Focus

This research specifically concentrates on the recreation and migration of the following four adversarial attack implementations:

1. adversarial-yolo
2. Adversatial_camou
3. Adversarial_Texture
4. Naturalistic_Patch_Attack

## Current Project Status

The project has been divided into two main phases: **Evaluation** and **Training (Attack Generation)**.

-   ✅ **Evaluation Phase: Complete**
    -   The necessary scripts to test a pre-trained adversarial patch or a perturbed image against a YOLOv5 model have been fully implemented.
    -   This includes loading the YOLOv5 model, pre-processing input images, applying the adversarial perturbation, and running inference to measure the attack's success (e.g., drop in mAP, misclassification rate).

-   ⌛ **Training Phase: In Progress**
    -   The core logic for *generating* the adversarial attacks (i.e., the training loop that optimizes the perturbation) is currently under development.
    -   This phase has proven to be significantly more complex than the evaluation phase due to fundamental architectural and codebase differences between YOLOv2 and YOLOv5. The key challenges encountered are detailed below.

## Key Challenges in Migrating the Training Pipeline

Migrating the training code for adversarial attacks from a relatively straightforward framework like YOLOv2 (Darknet) to the highly optimized and abstracted YOLOv5 (PyTorch) codebase presented several significant hurdles. The following are the most prominent issues I have been working to resolve:

### 1. Migrating the `max_prob_extractor` Logic

Many YOLOv2 attacks rely on a function to extract the maximum class probability score for a detected object within a patch region. This is used as the loss to minimize during the attack.

-   **The Problem**: The output format of YOLOv5 is fundamentally different from YOLOv2. YOLOv5 produces a tensor of shape `(batch_size, num_predictions, 5 + num_classes)` across multiple detection heads, where predictions are not yet filtered by Non-Max Suppression (NMS). The original `max_prob_extractor` from YOLOv2 codebases cannot be used.
-   **My Approach**: Replicating this required a deep dive into the YOLOv5 output format. I had to write a new utility function that parses the raw, multi-scale output from the model's detection heads. This function decodes the bounding boxes and class confidences, identifies which predictions fall within the adversarial patch's area-of-interest, and then finds the maximum confidence score associated with the target class. This was a non-trivial task that required carefully re-implementing parts of the NMS logic just for our loss calculation.

### 2. Needing to Bypass High-Level Ultralytics APIs

For simple inference, the Ultralytics API is excellent (`results = model(images)`). However, for generating white-box attacks, this level of abstraction is a hindrance.

-   **The Problem**: Generating an attack requires direct access to model internals—specifically, the raw outputs before post-processing (like NMS) and the ability to perform a backward pass based on those raw outputs. The high-level API hides these details completely.
-   **My Approach**: To gain the necessary control, I had to forgo the convenient API and interact directly with the `nn.Module` object. This involved creating a custom forward-pass function that calls `model.forward()` and works with the resulting list of tensors. This granular control is essential for establishing the end-to-end pipeline where I can inject a patch, get raw logits, calculate a custom loss, and backpropagate gradients back to the input patch.

### 3. Other Miscellaneous Issues with Training and Utility Functions.

## Installation

To set up the environment and run the evaluation scripts, please make sure you have YOLOv5+ cloned in your repository.

## Future Work

-   [ ] Finalize the migration of the training pipelines for all four attacks.
-   [ ] Conduct extensive experiments to generate new adversarial patches targeting YOLOv5+ models.
-   [ ] Evaluate the transferability of the newly generated patches across different YOLOv5+ model sizes (n, s, m, l, x).
-   [ ] Compare the results with the original papers to analyze the robustness of YOLOv5+ relative to YOLOv2.

