# A Mechanistic Analysis of Semantic Role Representation in a Decoder-Only Model

Investigating how `aya-expanse-8b` represents semantic roles (Agent vs Patient) using probing and activation patching.

All notebooks are designed to run on Google Colab with a T4 GPU and require a Hugging Face token for access to the gated model.

---

## Notebooks


### `layerwise_probing.ipynb`
Extracts hidden states for target nouns in simple active/passive sentences and trains a linear classifier (logistic regression) to predict Agent vs Patient. The probe is evaluated at every transformer layer to identify which layer best encodes semantic role information. The dataset (152 examples, 19 noun pairs) is designed so that nouns appear as both Agent and Patient, ruling out surface-level heuristics like grammatical function and word position.

### `positionwise_probing.ipynb`
Fixes the layer (layer 5) and instead varies the **token position** from which the hidden state is read: `noun1`, `verb`, `noun2`, `was`, `by`, and the `final` token. Each position is probed on the relevant subset of examples to see where in the sentence the role information is most accessible.

### `activation_patching.ipynb`
Performs node-level interchange intervention. Builds clean/corrupted sentence pairs that differ only in voice (active vs passive) and patches individual attention heads and MLPs at the final token position, one at a time. The normalized effect of each patch measures how much a component contributes to the model's ability to identify the correct Agent or Patient.

### `probing_patching_data.ipynb`
Applies layer-wise linear probing to the same sentence format used in the patching experiment (e.g. *"Alice warned Bob. The one who warned was"*). Reads the hidden state at the final `was` token and probes whether it encodes which name is the correct answer (e.g. *Alice)
