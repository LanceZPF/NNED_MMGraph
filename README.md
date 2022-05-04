# NNED_MMGraph

Neural Named Entity Disambiguation via MMGraph (together with SimTri).

This is an implementation of Self-supervised Enhancement for Named Entity Disambiguation via Multimodal Graph Convolution in Python. It needs pytorch libraries to be installed.

## Dataset

The data show be processed with `./ED_test/get_input.py` to be formed into pickle files. Then the train and test sets can be directly used.
The image data  used for the multimodal version of MMGraph and the training of SimTri can be downloaded in https://pan.baidu.com/s/1bfqJk_j6Jtpwukx6AIlq8w?pwd=csx5,
the extraction code is csx5.
The `image` dir contains the negative samples and the `image1` dir contains the negative samples.

Place the `data` dir into `./ED_test/`.

## Code

The UI is used for the naive test. Please see `./ED_test/README.md` for more details.
