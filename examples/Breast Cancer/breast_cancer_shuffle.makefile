# Original dataset acquired with standard licence from
# https://archive.ics.uci.edu/ml/machine-learning-databases/iris/

.decl
# Import each original IrisData data row as a whole
source_file CancerDataset = EXAMPLESROOT + "\Breast Cancer\Breast-cancer-dataset\breast-cancer-wisconsin.data"
{CancerData}

.res
# Instruct the interpreter that IrisData[i] is a string
CancerData: categorical

.act
new ShuffledData = shuffle(CancerData)

.sap
save_file SaveShuffled = EXAMPLESROOT + "\Breast Cancer\shuffled-cancer-dataset"
save "{ShuffledData}" into SaveShuffled

.make
