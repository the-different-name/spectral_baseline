# spectral_baseline
baseline subtraction algorithms including
  ALS-based (ALS, psalsa and derpsalsa),
  two morphological algorithms and 
  one wavelet-based.  

Random baseline and morph/pspline algorithm require the csaps package, which can be installed by the following command:

pip install -U csaps

Wavelet transform algorithm requires skued, which can be installed, depending on the python installation, by 

pip install scikit-ued

or

conda config –add channels conda-forge conda install scikit-ued

see https://scikit-ued.readthedocs.io/en/master/tutorials/baseline.html
