import numpy as np
import csv
import math
import random




with open('train.txt')as file:
    Inputs,Hidden,Output=[int(x) for x in file.readline().split()]
    examples=int(file.readline())
    arr=[]
    for line in file:
        arr.append(list(map(float,line.split())))
    data=np.array(arr)
    m=data[:,:Inputs]
    m=m/np.max(m,axis=0)   # normalization lldata 3shn a5ly l error oryb mn zero
    
    n=data[:,Inputs:Inputs+Output]
    n=n/np.max(n,axis=0)
    
    
    
  ########
def generateHweights(Hidden,Inputs):
    hiddenweights=[]
    for i in range(Inputs):  #lkol hidden node fyh connection m3 l inputs 
        connected=[]
        for j in range(Hidden):
            r=random.uniform(-10,10)
            connected.append(r)
        hiddenweights.append(connected)
    return hiddenweights   
hiddenweights=generateHweights(Hidden,Inputs)

def generateOweights(Hidden,Output):
    outputweights=[]
    for i in range(Hidden):
        connected=[]
        for j in range(Output):
            r=random.uniform(-10,10)
            connected.append(r)
        outputweights.append(connected)    
    return outputweights
outputweights=generateOweights(Hidden,Output)
    
def getsum(inputs,weights,n):
    summ=[]
    for i in range(n):
        temp=0
        for j in range(Inputs):
            temp+=inputs[j]*weights[j][i]
        summ.append(temp)    
    return summ
def sigmoid(summ):
    sigmoid=[]
    for i in range(len(summ)):
         try:   #3shn mytl3sh e 
             sigmoid.append(1 / (1 + math.exp(-summ[i])))
         except OverflowError:
             sigmoid.append(float('inf'))    
    return sigmoid
def feedforward(inputs):
    hsum=getsum(inputs,hiddenweights,Hidden)
    hsigmoid=sigmoid(hsum)
    osum=getsum(hsigmoid,outputweights,Output)
    osigmoid=sigmoid(osum)
    return osigmoid,hsigmoid
#b3den bn7sb el out put error w el sigma w b3den n3ml update ll weights
#actual output-calculated output(sigmoid)    
def errorforOutput(osigmoid,output):
    error=[]
    for i in range(len(output)):
        errors=output[i]-osigmoid[i]
        error.append(errors)
    return error
#using derivative of sigmoid 
#out(1-out)*error   bn7sbo lkol output 3ndna    
def sigmaOutput(error,osigmoid):
    sigma=[]
    for i in range(len(osigmoid)):
        temp=osigmoid[i]*(1-osigmoid[i])*error[i]
        sigma.append(temp)
    return sigma
# new =old+learning rate*sigma_of_output*hidden node value 
def updateOweights(sigma,hsigmoid):
    
    k=0
    j=0
    while(k<Output):
        while(j<Hidden):
            temp= outputweights[j][k]+(0.01*sigma[k]*hsigmoid[j])
            outputweights[j][k]=temp
            j+=1
        k+=1
#out_el7sbto_bl sigmoid(1-output)*(sigmaoutput*weight of node to the output)
#bnloop 3la kol node fl hidden layer w ashofha connected m3 loutputs  
        
def sigmaHidden (sigma,hsigmoid,outputweights,Hidden):
    Hsigma=[]
    for j in range (Hidden):
        temp=0
        for k in range(len(sigma)):
            temp+=sigma[k]*outputweights[j][k] 
        total=hsigmoid[j]*(1-hsigmoid[j])*temp 
        Hsigma.append(total)
    return Hsigma
#new weight=oldweight+(learningrate*sigma for the hidden nodewith corresponding input)
def updateHweights(Hsigma,inputvector,Hidden):
    for i in range(len(inputvector)):
        for j in range(Hidden):
            temp=hiddenweights[i][j]+(0.01*Hsigma[j]*inputvector[i])
            hiddenweights[i][j]=temp
    
def MSE (sigma):
    s=0
    for i in range(len(sigma)):
        s+=pow(sigma[i],2)
        
    return (s/2)

for i in range (300):
    for Epoch in range(examples):
        F=feedforward(m[Epoch])  #btl3 values for output and hidden
        Hiddenvalues=F[1]  #el hidden 
    
        Outputvalues=F[0]   #el output
        OutputError=errorforOutput(Outputvalues,n[Epoch])
        OutputSigma=sigmaOutput(n[Epoch],OutputError)
        avg=np.average(OutputSigma,axis=0)
        #get the average of each iteration output value 
        updateOweights(OutputSigma,Hiddenvalues)
        #update weights of output with hidden
        HiddenSigma=sigmaHidden(OutputError,Hiddenvalues,outputweights,Hidden)
        updateHweights(HiddenSigma,m[Epoch],Hidden)
        #update weights of hidden layer with input
        mse=MSE(OutputError)
        if mse <= avg:
            print("MSE: ",float(mse))
            break
    if mse <= avg:
        break
    if i==299:
        print("MSE: ",float(mse))




with open("Hweights.txt",'w') as f:
    for l in range(len(hiddenweights)):
        for j in range(len(hiddenweights[l])):
            f.write(str(hiddenweights[l][j]))
            f.write(" ")

        f.write('\n')
with open("Oweights.txt",'w') as f:
   for l in range(len(outputweights)):
        for j in range(len(outputweights[l])):
            f.write(str(outputweights[l][j]))
            f.write(" ")

        f.write('\n')
    


    

    
   
        