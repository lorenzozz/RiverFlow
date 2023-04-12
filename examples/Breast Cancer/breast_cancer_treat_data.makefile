.decl
source_file CancerData = EXAMPLESROOT + "/Breast Cancer/Breast-cancer-dataset/shuffled-cancer-dataset.data"
{Number},{Thickness},{CSize},{CShape},{MAdh},{SECS},{BNuclei},{BChrom},{NorNucleoli},{Mitoses},{Truth}
.res
Number: categorical
Thickness: numeric
CSize: numeric
CShape: numeric
MAdh: numeric
SECS: numeric
BNuclei: numeric
BChrom: numeric
NorNucleoli: numeric
Mitoses: numeric

# Save truth as categorical to use one hot encoding later..
Truth: categorical

.act

import numpy as np

Thickness = Thickness - media(Thickness)
CSize = CSize - media(CSize)
CShape = CShape - media(CShape)
MAdh = MAdh - media(MAdh)
SECS = SECS - media(SECS)
BNuclei = BNuclei - media(BNuclei)
BChrom = BChrom- media(BChrom)
NorNucleoli = NorNucleoli- media(NorNucleoli)
Mitoses = Mitoses - media(Mitoses)
new Indexes = np.arange(0, np.size(Thickness))
new OHotTruth = one_hot_encode(Truth, ["2", "4"])
print(OHotTruth)

.sap
.make

log_file LogFile = EXAMPLESROOT + "/Breast Cancer/Breast-cancer-dataset/log.txt"
plan_file Test = EXAMPLESROOT + "/Breast Cancer/Breast-cancer-dataset/Test"
plan_file Training = EXAMPLESROOT + "/Breast Cancer/Breast-cancer-dataset/Training"

begin plan CancerPred expecting attempt_recovery
{
    align Thickness, CSize, CShape, MAdh against Indexes,Indexes,Indexes,Indexes as index
    align SECS, BNuclei, BChrom, NorNucleoli, Mitoses against Indexes, Indexes, Indexes, Indexes, Indexes as index
    align OHotTruth against Indexes as index

    consider x
    take x from Thickness
    take x from CSize
    take x from CShape
    take x from MAdh
    take 2 before x from SECS
    take x from BNuclei
    take x from BChrom
    take x from NorNucleoli
    take x from Mitoses

    make OHotTruth the target and take y from OHotTruth
    pair x with target
}
end plan

split CancerPred into Test, Training as 30, 70
compile CancerPred
log CancerPred into LogFile
