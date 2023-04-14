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
new Res = load_vec(Value, ',')
new Self = np.arange(0, len(Res))
new Self2 = np.arange(0, len(Res))
new Self3 = np.arange(0, len(Res))

.sap


.make

plan_file NewFile = EXAMPLESROOT + '/River Height/savefile'
log_file Logs = EXAMPLESROOT + '/River Height/savefile'

begin plan NewPlan expecting attempt_recovery
{
    align Precipitazione against Data as date with format %Y-%M-%D
    align Res against DateTarg as date with format %Y-%M-%D
    consider x

    take 16 before x from Precipitazione

    make Precipitazione the target and take y from Precipitazione
    pair x and target
}
end plan

compile NewPlan into NewFile
log NewPlan into Logs
