.decl
source_file MeteoFile = EXAMPLESROOT + '/River Height/LOZZOLO_giornalieri_2001_2022pad.csv'
{Data};{PNove};{PZero};{TMedia};{TMax};{TMin};{Vel};{Raf};{Dur};{Set};{Temp}
.res

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

plot Raf

.sap

.make
