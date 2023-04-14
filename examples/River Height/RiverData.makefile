.decl
source_file RiverFile = RIVERDATAROOT + '/sesia-height.csv'
{Date};{Value}
.res

Date: categorical
Value: numeric
.act

import numpy as np

new v = np.array("[1,2,3]")
print(v)
new Var = media(Value)
new Var2 = dev_stand(Value)

.sap
.make
