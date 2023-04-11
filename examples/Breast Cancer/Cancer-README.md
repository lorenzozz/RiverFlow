The following example makes use of the UCI free-to-use dataset repository at https://archive.ics.uci.edu/ml/index.php. 
No ownership is claimed over the data presented.

After obtaining the .data and .names files from the ML Repository, it is necessary to treat them accordingly to their distribution.
The data is first shuffled in breast_cancer_shuffle.makefile as the result of the call to the shuffle(x) primitive. 

The data is read row-wise as a single block of information per row, shuffled and saved into shuffled-cancer-dataset.data.

After importing the shuffled data in breast_cancer_shuffle.makefile, all features but the categorical target labels are brought to have zero mean.

The target labels are then one-hot encoded with ordering "2","4".
It must be noted that in the original dataset the classification target classes, benign and malign, are numerically encoded using respectively 2 and 4.
We opted for a one-hot encoding for ease of use and better predictions from the model.

All features are then grouped in a single plan, containing one data point from each feature per total data point. The
one hot encoding of the target label is then used as target, which strongly suggests the need of a Cross-Entropy like loss function
for training.

Output logs and the resulting dataset are both present at /Breast Cancer/results.