.decl
source_file MeteoFile = EXAMPLESROOT + '/River Height/LOZZOLO_Dati.csv'
{Data};{Precipitazione};{TemperaturaMedia};{TemperaturaMassima};{TemperaturaMinima};{Velocita};{Raffica};{Durata};{Settore};{TempoPermanenza}

source_file RiverFile = EXAMPLESROOT + '/River Height/sesia-hourly-packed.csv'
{DateTarg};{Value}ì
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
new ResT = ResF
new Self = np.arange(0, len(ResF))
new Self2 = np.arange(0, len(ResF))
new Self3 = np.arange(0, len(ResF)) 
  
.sap




.make

plan_file NewFile = EXAMPLESROOT + '/River Height/savefile'
plan_file TestFile = EXAMPLESROOT + '/River Height/test'
log_file Logs = EXAMPLESROOT + '/River Height/savefile'

begin plan NewPlan expecting attempt_recovery
{
    align TemperaturaMassima against Data as date with format %Y-%M-%D
    align Precipitazione against Data as date with format %Y-%M-%D
    align ResT, ResF against DateTarg, DateTarg as date with format %Y-%M-%D
    align TemperaturaMedia against Data as date with format %Y-%M-%D
    consider x

    take 30 before x from TemperaturaMassima
    take 30 before x from TemperaturaMedia
    take 30 before x from ResF
    take 30 before x from Precipitazione

    make ResT the target and take y from ResT
    pair x and target
}
end plan

split NewPlan into NewFile, TestFile as 60, 40
compile NewPlan
log NewPlan into Logs
