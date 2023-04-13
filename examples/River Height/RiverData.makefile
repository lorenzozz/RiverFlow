.decl
source_file RiverFile = RIVERDATAROOT + '/sesia-height.csv'
{Date};{Value}
.res

Date: categorical
Value: numeric
.act

new Var = media(Value)
new Var2 = dev_stand(Value)
print(Var2)

.sap
.make
