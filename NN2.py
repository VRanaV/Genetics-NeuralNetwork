import csv
from NN1 import *

with open("Hweights.txt") as f:
    reader=csv.reader(f)
    h=list(reader)

HiddenW=[]

for i in range(len(h)):
    s=h[i][0].split()
    for j in range(len(s)):
        s[j]=float(s[j])
    HiddenW.append(s)



with open("Oweights.txt") as g:
    readerr=csv.reader(g)
    o=list(readerr)

OutputW=[]


for i in range(len(o)):
    s=o[i][0].split()
    for j in range(len(s)):
        s[j]=float(s[j])
    OutputW.append(s)
    
#mlyt el hidden wl output bl values lgdeda ml file    
hiddenweights=HiddenW
outputweights=OutputW
#loop 3l inputs
for i in range(len(m)):
    f=feedforward(m[i])
    O=f[0] #output values (sigmoid)
    Sigma=errorforOutput(O,n[i]) #n[i] -->actual outputs
    mse=MSE(Sigma)
    print('MSE: ',mse)




