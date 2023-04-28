.decl
source_file MeteoFile = EXAMPLESROOT + '/River Height/LOZZOLO_giornalieri_2001_2022pad.csv'
{Data};{PNove};{PZero};{TMedia};{TMax};{TMin};{Vel};{Raf};{Dur};{Set};{Temp}

source_file RiverFile = EXAMPLESROOT + '/River Height/sesia-hourly-packed-padded.csv'
{RData};{Vec}
.res

RData: categorical
Vec: categorical
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
new Total = np.hstack((list(ResF)))

plot Total
print(ResF)

.sap

.make
