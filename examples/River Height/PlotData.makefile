.decl
source_file MeteoFile = EXAMPLESROOT + '/River Height/LOZZOLO_giornalieri_2001_2022pad.csv'
{Data};{PNove};{PZero};{TMedia};{TMax};{TMin};{Vel};{Raf};{Dur};{Set};{Temp}

source_file RiverFile = EXAMPLESROOT + '/River Height/sesia-hourly-packed-padded-aligned.csv'
{RData};{Vec}
.res

# River
RData: categorical
Vec: categorical

# Meteo
Data: categorical
PNove: numeric
PZero: numeric
TMedia: numeric
TMax: numeric
TMin: numeric
Vel: numeric
Raf: numeric
Dur: numeric
Set: categorical
Temp: numeric

.act

import numpy as np
new ResF = load_vec(Vec, ',')
new ResT = ResF

new NormalizedRes = z_score(ResF)
new NormalizedResT = NormalizedRes
print(NormalizedRes)

.sap

.make

plan_file NewFile = EXAMPLESROOT + '/River Height/savefileN'
plan_file TestFile = EXAMPLESROOT + '/River Height/testN'
log_file Logs = EXAMPLESROOT + '/River Height/savefileN.txt'

begin plan NewPlan expecting attempt_recovery
{
#     align TMax against Data as date with format %Y-%M-%D
#     align Vel against Data as date with format %Y-%M-%D
#     align PZero, PNove against Data, Data as date with format %Y-%M-%D
    align NormalizedRes, NormalizedResT against RData, RData as date with format %Y-%M-%D
#     align TMedia against Data as date with format %Y-%M-%D
    align PNove against Data as date with format %Y-%M-%D
    consider x

#     take 7 before x and 14 after x from Vel
#     take 7 before x and 14 after x from TMax
#     take 7 before x and 14 after x from TMedia
    take 30 before x from NormalizedRes
    take 7 before x and 6 after x from PNove
#     take 7 before X and 14 after x from PNove

    make NormalizedResT the target and take 6 after y from NormalizedResT
    pair x and target
}
end plan

split NewPlan into NewFile, TestFile as 75, 25
compile NewPlan
log NewPlan into Logs
