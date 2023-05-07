.decl
source_file MeteoCellio = EXAMPLESROOT + '/Meteo/CELLIO_giornalieri_2006_2022Pad.csv'
{DataC};{PNoveC};{PZeroC};{TempAvgC};{TempMaxC};{TempMinC};

source_file MeteoCarcoforo = EXAMPLESROOT + '/Meteo/CARCOFORO_giornalieri_1996_2022Pad.csv'
{DataCa};{PNoveCa};{PZeroCa};{TempAvgCa};{TempMaxCa};{TempMinCa};

source_file MeteoLozzolo = EXAMPLESROOT + '/Meteo/LOZZOLO_giornalieri_2001_2022Pad.csv'
{DataLo};{PNoveLo};{PZeroLo};{TMediaLo};{TMaxLo};{TMinLo};{VelLo};{RafLo};{DurLo};{SetLo};{TempLo};

source_file MeteoRima = EXAMPLESROOT + '/Meteo/RIMA_giornalieri_2001_2022Pad.csv'
{DataRi};{PNoveRi};{PZeroRi};{TempAvgRi};{TempMaxRi};{TempMinRi};

source_file MeteoBocc = EXAMPLESROOT + '//Meteo/BOCCHETTA_DELLE_PISSE_giornalieri_1988_2022Pad.csv'
{DataBo};{PNoveBo};{PZeroBo};{NeveBo};{NeveSBo};{NeveAltBo};{TempAvgBo};{TempMaxBo};{TempMinBo};{VelMediaBo};{RafficaBo};{DurataBo};{RadBo};

source_file RiverFile = EXAMPLESROOT + '/River Height/sesia-hourly-packed-padded-aligned.csv'
{RData};{VectorRH}
.res

# ************ Fiume ************
RData: categorical
VectorRH: categorical

# ************ Meteo ************

# > Date di ogni meteo
# !NOT PRESENT IN MODEL!
DataC: categorical
DataCa: categorical
DataBo: categorical
DataRi: categorical
DataLo: categorical

# > Pioggia di ogni meteo

PNoveC: numeric
PNoveCa: numeric
PNoveLo: numeric
PNoveRi: numeric
PNoveBo: numeric

PZeroC: numeric
PZeroCa: numeric
PZeroLo: numeric
PZeroRi: numeric
PZeroBo: numeric

# > Temperatura di ogni meteo

TempAvgC: numeric
TempMaxC: numeric
TempMinC: numeric
TempMinCa: numeric
TempMaxCa: numeric
TempAvgCa: numeric
TMediaLo: numeric
TMinLo: numeric
TMaxLo: numeric
TempMaxRi: numeric
TempMinRi: numeric
TempAvgRi: numeric
TempMinBo: numeric
TempMaxBo: numeric
TempAvgBo: numeric

# > Vento

VelLo: numeric
RafLo: numeric
DurLo: numeric
TempLo: numeric
# !NOT PRESENT IN MODEL!
SetLo: categorical
# ^^^^^^^
RafficaBo: numeric
DurataBo: numeric

# > Varie

RadBo: numeric
NeveBo: numeric
NeveSBo: numeric
NeveAltBo: numeric
VelMediaBo: numeric

.act

VectorRH = load_vec(VectorRH, sep=',')

VectorRH = z_score(VectorRH)
PNoveC = z_score(PNoveC)
PZeroC = z_score(PZeroC)
TempMaxC = z_score(TempMaxC)
TempMinC = z_score(TempMinC)
TempAvgC = z_score(TempAvgC)

PNoveCa = z_score(PNoveCa)
PZeroCa = z_score(PZeroCa)
TempAvgCa = z_score(TempAvgCa)
TempMaxCa = z_score(TempMaxCa)
TempMinCa = z_score(TempMinCa)

PNoveLo = z_score(PNoveLo)
PZeroLo = z_score(PZeroLo)
TMediaLo = z_score(TMediaLo)
TMaxLo = z_score(TMaxLo)
TMinLo = z_score(TMinLo)
VelLo = z_score(VelLo)
RafLo = z_score(RafLo)
DurLo = z_score(DurLo)
TempLo = z_score(TempLo)

PNoveRi = z_score(PNoveRi)
PZeroRi = z_score(PZeroRi)
TempAvgRi = z_score(TempAvgRi)
TempMaxRi = z_score(TempMaxRi)
TempMinRi = z_score(TempMinRi)

PNoveBo = z_score(PNoveBo)
PZeroBo = z_score(PZeroBo)
NeveBo = z_score(NeveBo)
NeveSBo = z_score(NeveSBo)
NeveAltBo = z_score(NeveAltBo)
TempAvgBo = z_score(TempAvgBo)
TempMaxBo = z_score(TempMaxBo)
TempMinBo = z_score(TempMinBo)
VelMediaBo = z_score(VelMediaBo)
RafficaBo = z_score(RafficaBo)
DurataBo = z_score(DurataBo)
RadBo = z_score(RadBo)

print(DurataBo)


new VectorRHCopy = VectorRH
.sap

.make

plan_file NewFile = EXAMPLESROOT + '/River Height/TestMeteo'
plan_file TestFile = EXAMPLESROOT + '/River Height/TestMeteoTest'
log_file Logs = EXAMPLESROOT + '/River Height/TestMeteoLog.txt'

begin plan RNND expecting attempt_recovery
{
    align PNoveC, PZeroC, TempAvgC, TempMaxC, TempMinC against DataC, DataC, DataC, DataC, DataC as date with format %Y-%M-%D
    align PNoveCa, PZeroCa, TempAvgCa, TempMaxCa, TempMinCa against DataCa, DataCa, DataCa, DataCa, DataCa as date with format %Y-%M-%D
    align PNoveLo, PZeroLo, TMediaLo, TMaxLo, TMinLo, VelLo, RafLo, DurLo, TempLo against DataLo, DataLo, DataLo, DataLo, DataLo, DataLo, DataLo, DataLo, DataLo as date with format %Y-%M-%D
    align PNoveRi, PZeroRi, TempAvgRi, TempMaxRi, TempMinRi against DataRi, DataRi, DataRi, DataRi, DataRi as date with format %Y-%M-%D
    align PNoveBo, PZeroBo, NeveBo, NeveSBo, NeveAltBo, TempAvgBo, TempMaxBo, TempMinBo, VelMediaBo, RafficaBo, DurataBo, RadBo against DataBo, DataBo, DataBo, DataBo, DataBo, DataBo, DataBo, DataBo, DataBo, DataBo, DataBo, DataBo as date with format %Y-%M-%D
    align VectorRH, VectorRHCopy against RData, RData as date with format %Y-%M-%D

    consider x
    take 6 before x and 6 after x from PNoveC
    take 6 before x and 6 after x from PZeroC
    take 6 before x and 6 after x from TempAvgC
    take 6 before x and 6 after x from TempMaxC
    take 6 before x and 6 after x from TempMinC
    take 6 before x and 6 after x from PNoveCa
    take 6 before x and 6 after x from PZeroCa
    take 6 before x and 6 after x from TempAvgCa
    take 6 before x and 6 after x from TempMaxCa
    take 6 before x and 6 after x from TempMinCa
    take 6 before x and 6 after x from PNoveLo
    take 6 before x and 6 after x from PZeroLo
    take 6 before x and 6 after x from TMediaLo
    take 6 before x and 6 after x from TMaxLo
    take 6 before x and 6 after x from TMinLo
    take 6 before x and 6 after x from VelLo
    take 6 before x and 6 after x from RafLo
    take 6 before x and 6 after x from DurLo
    take 6 before x and 6 after x from TempLo
    take 6 before x and 6 after x from PNoveRi
    take 6 before x and 6 after x from PZeroRi
    take 6 before x and 6 after x from TempAvgRi
    take 6 before x and 6 after x from TempMaxRi
    take 6 before x and 6 after x from TempMinRi
    take 6 before x and 6 after x from PNoveBo
    take 6 before x and 6 after x from PZeroBo
    take 6 before x and 6 after x from PNoveBo
    take 6 before x and 6 after x from NeveBo
    take 6 before x and 6 after x from NeveSBo
    take 6 before x and 6 after x from NeveAltBo
    take 6 before x and 6 after x from TempAvgBo
    take 6 before x and 6 after x from TempMaxBo
    take 6 before x and 6 after x from TempMinBo
    take 6 before x and 6 after x from VelMediaBo
    take 6 before x and 6 after x from RafficaBo
    take 6 before x and 6 after x from DurataBo
    take 6 before x and 6 after x from RadBo
    take 12 before x from VectorRH


    make VectorRHCopy the target and take 7 after y from VectorRHCopy
}
end plan

split RNND into NewFile, TestFile as 75, 25
compile RNND
log RNND into Logs

# begin plan NewPlan expecting attempt_recovery
# {
#     align TMax against Data as date with format %Y-%M-%D
#     align Vel against Data as date with format %Y-%M-%D
#     align PZero, PNove against Data, Data as date with format %Y-%M-%D
#    align NormalizedRes, NormalizedResT against RData, RData as date with format %Y-%M-%D
#     align TMedia against Data as date with format %Y-%M-%D
#    align PNove against Data as date with format %Y-%M-%D
#    consider x

#     take 7 before x and 14 after x from Vel
#     take 7 before x and 14 after x from TMax
#     take 7 before x and 14 after x from TMedia
#    take 30 before x from NormalizedRes
#    take 7 before x and 6 after x from PNove
#     take 7 before X and 14 after x from PNove

#    make NormalizedResT the target and take 6 after y from NormalizedResT
#    pair x and target
#}
#end plan

#split NewPlan into NewFile, TestFile as 75, 25
#compile NewPlan
#log NewPlan into Logs
