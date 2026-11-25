# Sparse Digit Classification from Sufficient Statistics

Experiments on sparse least-squares classifiers for handwritten-digit parity, with five optimization algorithms and paired label-privacy controls.

Run these commands in the extracted folder using Python 3.12. The benchmark used NumPy 2.3.5 on an arm64 Mac with Apple Accelerate and a single requested BLAS thread.

~~~sh
python -m pip install -r requirements.txt
python experiment.py
python mnist_replication.py
python privacy.py
python diagnostics.py
~~~

**Data.** The Optdigits input is bundled. The MNIST scripts share a cache of four checksum-verified IDX archives downloaded from the CVDF mirror at first use (about 12 MB). Both accept --cache to select an existing directory. The public MNIST experiments use the complete official 60,000/10,000 split.

**Outputs.** The experiment scripts write to results/, mnist/, and privacy/ respectively. These directories already contain the measured data used in the report. Pass --out followed by a new directory to preserve those records during a new run. The diagnostics script reads the default directories and calculates the cross-split Jaccard scores and stopping-time comparisons.

**Figures.** After installing ReportLab 4.4.9, run the following in the same directory.

~~~sh
python -m pip install reportlab==4.4.9
python plots.py
python privacy_plot.py
~~~

**Methods.** The common stopping criterion for the five-solver benchmark is an absolute primal–dual gap of 1e-8. Seven timed trials follow one warm-up for each solver and problem. Timing depends on hardware. In the label-privacy simulation a fixed absolute penalty of 0.06 is used and the perturbed objective is solved to a KKT residual of 1e-9. Fixed simulation noise allows a paired comparison of the joint mechanism and the two component controls on public data.

**Data credits.** E. Alpaydin and C. Kaynak (1998), [Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49). The unmodified optdigits.tes partition is included under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Y. LeCun, L. Bottou, Y. Bengio and P. Haffner (1998), [Gradient-based learning applied to document recognition](https://doi.org/10.1109/5.726791). The MNIST loader retrieves the original files from the [CVDF mirror](https://github.com/cvdfoundation/mnist).
