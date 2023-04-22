.decl
source_file MeteoFile = EXAMPLESROOT + '/River Height/LOZZOLO_Dati.csv'
{Data};{Precipitazione};{TemperaturaMedia};{TemperaturaMassima};{TemperaturaMinima};{Velocita};{Raffica};{Durata};{Settore};{TempoPermanenza}

source_file RiverFile = EXAMPLESROOT + '/River Height/sesia-hourly-packed.csv'
{DateTarg};{Value}
.res

Data: categorical
Precipitazione: numeric
TemperaturaMedia: numeric
TemperaturaMassima: numeric
TemperaturaMinima: numeric
Velocita: numeric
Raffica: numeric
Durata: numeric
Settore: categorical
TempoPermanenza: numeric

DateTarg: categorical
Value: categorical

.act

import numpy as np
new ResF = load_vec(Value, ',')

ResF = max_linear(ResF, True)
TemperaturaMassima = max_linear(TemperaturaMassima, True)
Precipitazione = max_linear(Precipitazione, True)
TemperaturaMedia = max_linear(TemperaturaMedia, True)

new ResT = ResF

.sap


.make

plan_file NewFile = EXAMPLESROOT + '/River Height/savefileM'
plan_file TestFile = EXAMPLESROOT + '/River Height/testM'
log_file Logs = EXAMPLESROOT + '/River Height/savefileM'

begin plan NewPlan expecting attempt_recovery
{
    align TemperaturaMassima against Data as date with format %Y-%M-%D
    align Precipitazione against Data as date with format %Y-%M-%D
    align ResT, ResF against DateTarg, DateTarg as date with format %Y-%M-%D
    align TemperaturaMedia against Data as date with format %Y-%M-%D
    consider x

    take 30 before x and 14 after x from Precipitazione
    take 30 before x from TemperaturaMassima
    take 30 before x from TemperaturaMedia
    take 30 before x from ResF

    make ResT the target and take 5 after y from ResT
    pair x and target
}
end plan

split NewPlan into NewFile, TestFile as 60,40
compile NewPlan
log NewPlan into Logs
