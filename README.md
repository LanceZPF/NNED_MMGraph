# NNED_MMGraph

Neural Named Entity Disambiguation via MMGraph (together with SimTri).

This is an implementation of Self-supervised Enhancement for Named Entity Disambiguation via Multimodal Graph Convolution in Python. It needs pytorch libraries to be installed.

## Dataset

The data show be precessed with `get_input.py` to be formed into pickle files. Then the train and test sets can be directly used.
The image dir contains the possitive samples
The image1 dir contains the negative samples
The images are used for the multimodal version of MMGraph and the training of SimTri.

## Code

The UI is used for the naive test. See `train.py` for training.
