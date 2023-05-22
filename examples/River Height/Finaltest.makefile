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

source_file MeteoAlagna = EXAMPLESROOT+'/Meteo/ALAGNA_giornalieriPad.csv'
{DataAl};{PNoveAl};{PZeroAl};{NeveFrescaAl};{Neve2Al};{Neve3Al};{TempAvgAl};{TempMaxAl};{TempMinAl};

source_file MeteoRassa = EXAMPLESROOT+'/Meteo/RASSA_giornalieriPad.csv'
{DataRSDebug};{PNoveRass};{PZeroRass};{TempAvgRass};{TempMaxRass};{TempMinRass};{HMedRass};{HMinRass};{HMaxRass};

source_file MeteoBoccl = EXAMPLESROOT+'/Meteo/BOCCIOLO_giornalieriPad.csv'
{DataBoccl};{PNoveBoccl};{PZeroBoccl};{TempAvgBoccl};{TempMaxBoccl};{TempMinBoccl};{HMedBoccl};{HMinBoccl};{HMaxBoccl};


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
DataAl: categorical
DataRSDebug: categorical
DataBoccl: categorical

# > Pioggia di ogni meteo

PNoveC: numeric
PNoveCa: numeric
PNoveLo: numeric
PNoveRi: numeric
PNoveBo: numeric
PNoveAl: numeric
PNoveRass: numeric
PNoveBoccl: numeric

PZeroC: numeric
PZeroCa: numeric
PZeroLo: numeric
PZeroRi: numeric
PZeroBo: numeric
PZeroAl: numeric
PZeroRass: numeric
PZeroBoccl: numeric

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
TempMinAl: numeric
TempMaxAl: numeric
TempAvgAl: numeric
TempMinBoccl: numeric
TempMaxBoccl: numeric
TempAvgBoccl: numeric
TempMinRass: numeric
TempMaxRass: numeric
TempAvgRass: numeric

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
NeveFrescaAl: numeric
Neve2Al: numeric
Neve3Al: numeric
HMedBoccl: numeric
HMinBoccl: numeric
HMaxBoccl: numeric
HMedRass: numeric
HMinRass: numeric
HMaxRass: numeric

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

PNoveAl = z_score(PNoveAl)
PZeroAl = z_score(PZeroAl)
NeveFrescaAl = z_score(NeveFrescaAl)
TempAvgAl = z_score(TempAvgAl)
TempMaxAl = z_score(TempMaxAl)
TempMinAl = z_score(TempMinAl)

PNoveRass = z_score(PNoveRass)
PZeroRass = z_score(PZeroRass)
TempAvgRass = z_score(TempAvgRass)
TempMaxRass = z_score(TempMaxRass)
TempMinRass = z_score(TempMinRass)
PNoveBoccl = z_score(PNoveBoccl)
PZeroBoccl = z_score(PZeroBoccl)
TempAvgBoccl = z_score(TempAvgBoccl)
TempMaxBoccl = z_score(TempMaxBoccl)
TempMinBoccl = z_score(TempMinBoccl)
HMedBoccl = z_score(HMedBoccl)

new PNoveCCopy = PNoveC
new PZeroCCopy = PZeroC
new TempAvgCCopy = TempAvgC
new TempMaxCCopy = TempMaxC
new TempMinCCopy = TempMinC
new PNoveCaCopy = PNoveCa
new PZeroCaCopy = PZeroCa
new TempAvgCaCopy = TempAvgCa
new TempMaxCaCopy = TempMaxCa
new TempMinCaCopy = TempMinCa
new PNoveLoCopy = PNoveLo
new PZeroLoCopy = PZeroLo
new TMediaLoCopy = TMediaLo
new TMaxLoCopy = TMaxLo
new TMinLoCopy = TMinLo
new VelLoCopy = VelLo
new RafLoCopy = RafLo
new DurLoCopy = DurLo
new TempLoCopy = TempLo
new PNoveRiCopy = PNoveRi
new PZeroRiCopy = PZeroRi
new TempAvgRiCopy = TempAvgRi
new TempMaxRiCopy = TempMaxRi
new TempMinRiCopy = TempMinRi
new PNoveBoCopy = PNoveBo
new PZeroBoCopy = PZeroBo
new PNoveBoCopy = PNoveBo
new NeveBoCopy = NeveBo
new NeveSBoCopy = NeveSBo
new NeveAltBoCopy = NeveAltBo
new TempAvgBoCopy = TempAvgBo
new TempMaxBoCopy = TempMaxBo
new TempMinBoCopy = TempMinBo

new PZeroAlCopy = PZeroAl
new PNoveAlCopy = PNoveAl
new TempAvgAlCopy = TempAvgAl
new TempMaxAlCopy = TempMaxAl
new TempMinAlCopy = TempMinAl

new PZeroBocclCopy = PZeroBoccl
new PNoveBocclCopy = PNoveBoccl
new TempAvgBocclCopy = TempAvgBoccl
new TempMaxBocclCopy = TempMaxBoccl
new TempMinBocclCopy = TempMinBoccl

new PZeroRassCopy = PZeroRass
new PNoveRassCopy = PNoveRass
new TempAvgRassCopy = TempAvgRass
new TempMaxRassCopy = TempMaxRass
new TempMinRassCopy = TempMinRass

new VelMediaBoCopy = VelMediaBo
new RafficaBoCopy = RafficaBo
new DurataBoCopy = DurataBo
new RadBoCopy = RadBo
new HMedRassCopy = HMedRass
new HMinRassCopy = HMinRass
new HMaxRassCopy = HMaxRass
new HMedBocclCopy = HMedBoccl
new HMinBocclCopy = HMinBoccl
new HMaxBocclCopy = HMaxBoccl

new NeveFrescaAlCopy = NeveFrescaAl



new VectorRHCopy = VectorRH

.sap

.make

plan_file NewFile = EXAMPLESROOT + '/River Height/CompleteModelTraining'
plan_file TestFile = EXAMPLESROOT + '/River Height/CompleteModelTest'
log_file Logs = EXAMPLESROOT + '/River Height/CompleteModelLog.txt'

begin plan RNND expecting attempt_recovery
{
    align PNoveC, PZeroC, TempAvgC, TempMaxC, TempMinC against DataC, DataC, DataC, DataC, DataC as date with format %Y-%M-%D
    align PNoveCa, PZeroCa, TempAvgCa, TempMaxCa, TempMinCa against DataCa, DataCa, DataCa, DataCa, DataCa as date with format %Y-%M-%D
    align PNoveLo, PZeroLo, TMediaLo, TMaxLo, TMinLo, VelLo against DataLo, DataLo, DataLo, DataLo, DataLo, DataLo as date with format %Y-%M-%D
    align PNoveRi, PZeroRi, TempAvgRi, TempMaxRi, TempMinRi against DataRi, DataRi, DataRi, DataRi, DataRi as date with format %Y-%M-%D
    align PNoveBo, PZeroBo, NeveBo, TempAvgBo, TempMaxBo, TempMinBo, VelMediaBo against DataBo, DataBo, DataBo, DataBo, DataBo, DataBo, DataBo as date with format %Y-%M-%D
    align PNoveAl, PZeroAl, NeveFrescaAl, TempAvgAl, TempMaxAl, TempMinAl against DataAl, DataAl, DataAl, DataAl, DataAl, DataAl as date with format %Y-%M-%D
    align PNoveRass, PZeroRass, TempAvgRass, TempMaxRass, TempMinRass against DataRSDebug, DataRSDebug, DataRSDebug, DataRSDebug, DataRSDebug as date with format %Y-%M-%D
    align PNoveBoccl, PZeroBoccl, TempAvgBoccl, TempMaxBoccl, TempMinBoccl, HMedBoccl against DataBoccl, DataBoccl, DataBoccl, DataBoccl, DataBoccl, DataBoccl as date with format %Y-%M-%D

    align PNoveCCopy, PZeroCCopy, TempAvgCCopy, TempMaxCCopy, TempMinCCopy against DataC, DataC, DataC, DataC, DataC as date with format %Y-%M-%D
    align PNoveCaCopy, PZeroCaCopy, TempAvgCaCopy, TempMaxCaCopy, TempMinCaCopy against DataCa, DataCa, DataCa, DataCa, DataCa as date with format %Y-%M-%D
    align PNoveLoCopy, PZeroLoCopy, TMediaLoCopy, TMaxLoCopy, TMinLoCopy, VelLoCopy against DataLo, DataLo, DataLo, DataLo, DataLo, DataLo  as date with format %Y-%M-%D
    align PNoveRiCopy, PZeroRiCopy, TempAvgRiCopy, TempMaxRiCopy, TempMinRiCopy against DataRi, DataRi, DataRi, DataRi, DataRi as date with format %Y-%M-%D
    align PNoveBoCopy, PZeroBoCopy, NeveBoCopy, TempAvgBoCopy, TempMaxBoCopy, TempMinBoCopy, VelMediaBoCopy against DataBo, DataBo, DataBo, DataBo, DataBo, DataBo, DataBo as date with format %Y-%M-%D
    align PNoveAlCopy, PZeroAlCopy, NeveFrescaAlCopy, TempAvgAlCopy, TempMaxAlCopy, TempMinAlCopy against DataAl, DataAl, DataAl, DataAl, DataAl, DataAl as date with format %Y-%M-%D
    align PNoveRassCopy, PZeroRassCopy, TempAvgRassCopy, TempMaxRassCopy, TempMinRassCopy against DataRSDebug, DataRSDebug, DataRSDebug, DataRSDebug, DataRSDebug as date with format %Y-%M-%D
    align PNoveBocclCopy, PZeroBocclCopy, TempAvgBocclCopy, TempMaxBocclCopy, TempMinBocclCopy, HMedBocclCopy against DataBoccl, DataBoccl, DataBoccl, DataBoccl, DataBoccl, DataBoccl as date with format %Y-%M-%D
    align VectorRH, VectorRHCopy against RData, RData as date with format %Y-%M-%D

    consider x
    take 6 before x from PNoveC
    take 6 before x from PZeroC
    take 6 before x from TempAvgC
    take 6 before x from TempMaxC
    take 6 before x from TempMinC

    take 6 before x from PNoveCa
    take 6 before x from PZeroCa
    take 6 before x from TempAvgCa
    take 6 before x from TempMaxCa
    take 6 before x from TempMinCa

    take 6 before x from PNoveLo
    take 6 before x from PZeroLo
    take 6 before x from TMediaLo
    take 6 before x from TMaxLo
    take 6 before x from TMinLo
    take 6 before x from VelLo

    take 6 before x from PNoveRi
    take 6 before x from PZeroRi
    take 6 before x from TempAvgRi
    take 6 before x from TempMaxRi
    take 6 before x from TempMinRi

    take 6 before x from PZeroBo
    take 6 before x from PNoveBo
    take 6 before x from NeveBo
    take 6 before x from TempAvgBo
    take 6 before x from TempMaxBo
    take 6 before x from TempMinBo
    take 6 before x from VelMediaBo

    take 6 before x from PNoveAl
    take 6 before x from PZeroAl
    take 6 before x from NeveFrescaAl
    take 6 before x from TempAvgAl
    take 6 before x from TempMaxAl
    take 6 before x from TempMinAl

    take 6 before x from PNoveRass
    take 6 before x from PZeroRass
    take 6 before x from TempAvgRass
    take 6 before x from TempMaxRass
    take 6 before x from TempMinRass

    take 6 before x from PNoveBoccl
    take 6 before x from PZeroBoccl
    take 6 before x from TempAvgBoccl
    take 6 before x from TempMaxBoccl
    take 6 before x from TempMinBoccl
    take 6 before x from HMedBoccl

    take 6 before x from VectorRH


    make PNoveCCopy the target and take 6 after y from PNoveCCopy
    make PZeroCCopy the target and take 6 after y from PZeroCCopy
    make TempAvgCCopy the target and take 6 after y from TempAvgCCopy
    make TempMaxCCopy the target and take 6 after y from TempMaxCCopy
    make TempMinCCopy the target and take 6 after y from TempMinCCopy

    make PNoveCaCopy the target and take 6 after y from PNoveCaCopy
    make PZeroCaCopy the target and take 6 after y from PZeroCaCopy
    make TempAvgCaCopy the target and take 6 after y from TempAvgCaCopy
    make TempMaxCaCopy the target and take 6 after y from TempMaxCaCopy
    make TempMinCaCopy the target and take 6 after y from TempMinCaCopy

    make PNoveLoCopy the target and take 6 after y from PNoveLoCopy
    make PZeroLoCopy the target and take 6 after y from PZeroLoCopy
    make TMediaLoCopy the target and take 6 after y from TMediaLoCopy
    make TMaxLoCopy the target and take 6 after y from TMaxLoCopy
    make TMinLoCopy the target and take 6 after y from TMinLoCopy
    make VelLoCopy the target and take 6 after y from VelLoCopy

    make PNoveRiCopy the target and take 6 after y from PNoveRiCopy
    make PZeroRiCopy the target and take 6 after y from PZeroRiCopy
    make TempAvgRiCopy the target and take 6 after y from TempAvgRiCopy
    make TempMaxRiCopy the target and take 6 after y from TempMaxRiCopy
    make TempMinRiCopy the target and take 6 after y from TempMinRiCopy

    make PZeroBoCopy the target and take 6 after y from PZeroBoCopy
    make PNoveBoCopy the target and take 6 after y from PNoveBoCopy
    make NeveBoCopy the target and take 6 after y from NeveBoCopy
    make TempAvgBoCopy the target and take 6 after y from TempAvgBoCopy
    make TempMaxBoCopy the target and take 6 after y from TempMaxBoCopy
    make TempMinBoCopy the target and take 6 after y from TempMinBoCopy
    make VelMediaBoCopy the target and take 6 after y from VelMediaBoCopy

    make PNoveAlCopy the target and take 6 after y from PNoveAlCopy
    make PZeroAlCopy the target and take 6 after y from PZeroAlCopy
    make NeveFrescaAlCopy the target and take 6 after y from NeveFrescaAlCopy
    make TempAvgAlCopy the target and take 6 after y from TempAvgAlCopy
    make TempMaxAlCopy the target and take 6 after y from TempMaxAlCopy
    make TempMinAlCopy the target and take 6 after y from TempMinAlCopy

    make PNoveRassCopy the target and take 6 after y from PNoveRassCopy
    make PZeroRassCopy the target and take 6 after y from PZeroRassCopy
    make TempAvgRassCopy the target and take 6 after y from TempAvgRassCopy
    make TempMaxRassCopy the target and take 6 after y from TempMaxRassCopy
    make TempMinRassCopy the target and take 6 after y from TempMinRassCopy

    make PNoveBocclCopy the target and take 6 after y from PNoveBocclCopy
    make PZeroBocclCopy the target and take 6 after y from PZeroBocclCopy
    make TempAvgBocclCopy the target and take 6 after y from TempAvgBocclCopy
    make TempMaxBocclCopy the target and take 6 after y from TempMaxBocclCopy
    make TempMinBocclCopy the target and take 6 after y from TempMinBocclCopy
    make HMedBocclCopy the target and take 6 after y from HMedBocclCopy

    make VectorRHCopy the target and take 6 after y from VectorRHCopy

}
end plan

split RNND into NewFile, TestFile as 75, 25
compile RNND
log RNND into Logs

