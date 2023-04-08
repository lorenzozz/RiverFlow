# Original dataset acquired with standard licence from
# https://archive.ics.uci.edu/ml/machine-learning-databases/iris/

.decl
# Import shuffled iris data
source_file IrisData = "Your/Shuffled/Data/File"
{SLength},{SWidth},{PLength},{PWidth},{Category}

# SLength stands for sepal Length
# SWidth stands for sepal width
# PLength stands for petal length
# PWidth stands for petal width

.res
SLength: numeric
SWidth: numeric
PLength: numeric
PWidth: numeric
Category: categorical

.act
import numpy as np

new length = np.size(SLength)
# Use Indexes to align data in plan
new Indexes = np.arange(0, length)

# Scale data to have zero mean

SLength = SLength - media(SLength)
SWidth = SWidth - media(SWidth)
PLength = PLength - media(PLength)
PWidth = PWidth - media(PWidth)

new Categories = da_categorico_a_numero(Category, {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

.sap

save_file SaveScaled = "/Save/path"
save "{SLength},{SWidth},{PLength},{PWidth}" into SaveScaled

.make

plan_file Training = "Training/Path/File"
plan_file Test = "Test/Path/File"
log_file Logs = "Log/Path/File"

begin plan IrisPlan expecting full_recovery
{
	align SLength, SWidth, PLength, PWidth against Indexes, Indexes, Indexes, Indexes as index
    	align Categories against Indexes as index

	consider x
	take x from SLength
	take x from SWidth
	take x from PLength
	take x from PWidth

	make Categories the target and take y from Categories
	pair x with target

}
end plan

# Dividi il dataset in due sezioni, una di training e una di test
split IrisPlan into Training, Test as 70, 30

compile IrisPlan
log IrisPlan into Logs
