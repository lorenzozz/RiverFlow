# Original dataset acquired with standard licence from
# https://archive.ics.uci.edu/ml/machine-learning-databases/iris/

.decl
# Import each original IrisData data row as a whole
source_file IrisData = "Your/Shuffled/Data/File"
{IrisData}

.res
# Instruct the interpreter that IrisData[i] is a string
IrisData: categorical

.act
new ShuffledData = shuffle(IrisData)

.sap
save_file SaveShuffled = "/your/save/path"
save "{ShuffledData}" into SaveShuffled

.make