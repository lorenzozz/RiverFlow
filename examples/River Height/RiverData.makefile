.decl
source_file RiverFile = EXAMPLESROOT + '/River Height/sesia-hourly-packed.csv'
{Date};{Value}
.res

Date: categorical
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
    align Res against Self as index
    align Self2 against Self as index
    align Self3 against Self as index
    consider x

    take 2 after x from Self3
    take 2 before x from Res
    make Self2 the target and take y from Self2
    pair x and target

}
end plan

compile NewPlan into NewFile
log NewPlan into Logs
