#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#Given X
x_1_1=[[7.4]]
x_1_2=[[7.2]]
x_1_3=[[7.01]]
x_1_4=[[7.8]]
x_1_5=[[7.88]]
x_2_1=[[2.6]]
x_2_2=[[2.7]]
x_2_3=[[2.4]]
x_2_4=[[2.8]]
x_2_5=[[3]]
x_3_1=[[8,90]]
x_3_2=[[1,40]]
x_3_3=[[15,68.69]]
x_3_4=[[10,20]]


# In[3]:


def functions(problem,x,theta):
    if problem==1:
        return 9*x[0]-4*np.log(x[0]-7)
    if problem==2:
        return 6*x[0]-4*np.log(x[0]-2)-3*np.log(25-x[0])
    if problem==3:
        return -9*x[0]-10*x[1]+theta*(-np.log(100-x[0]-x[1])-np.log(x[0])-np.log(x[1])-np.log(50-x[0]+x[1]))


# In[4]:


def gradientF(problem,x,theta):
    if problem==1:
        return 9-4/(x[0]-7)
    if problem==2:
        return 6-4/(x[0]-2)+3/(25-x[0])
    if problem==3:
        return np.array([
            [-9+theta*(1/(100-x[0]-x[1])-1/x[0]+1/(50-x[0]+x[1]))],
            [-10+theta*(1/(100-x[0]-x[1])-1/x[1]-1/(50-x[0]+x[1]))]
        ])


# In[5]:


def hessianF(problem,x,theta):
    if problem==1:
        return 4/np.power(x[0]-7,2)
    if problem==2:
        return 4/np.power(x[0]-2,2)-3/np.power(25-x[0],2)
    if problem==3:
        return np.array([
            [theta*(-1/np.power(100-x[0]-x[1],2)-1/np.power(x[0],2)-1/np.power(50-x[0]+x[1],2)), theta*(-1/np.power(100-x[0]-x[1],2)+1/np.power(50-x[0]+x[1],2))],
            [theta*(-1/np.power(100-x[0]-x[1],2)+1/np.power(50-x[0]+x[1],2)), theta*(-1/np.power(100-x[0]-x[1],2)-1/np.power(x[1],2)-1/np.power(50-x[0]+x[1],2))]
        ])


# In[6]:


def compute_d(problem,x,theta):
    if problem==1:
        return -gradientF(problem,x,theta)/hessianF(problem,x,theta)
    if problem==2:
        return -gradientF(problem,x,theta)/hessianF(problem,x,theta)
    if problem==3:
        return -np.linalg.inv(hessianF(problem,x,theta)).dot(gradientF(problem,x,theta))


# In[7]:


#print table
def printTable(X):
    print('\\begin{center}')
    print('\\begin{tabular}{|| c c ||} ')
    print('\\hline')
    print('k & $x^k$ \\\\')
    print('\\hline\\hline')
    for i in range(len(X)):
        print('%d & %d \\\\ '%(i,X[i][0]))
        print('\\hline')
    print('\\end{tabular}')
    print('\\end{center}')
    


# In[8]:


def NewtonsMethod(problem,X,alpha,theta,nmax):
    k=0
    flag=1
    d_k=[]
    dim=len(X[k])
    alpha_k=[]
    for i in range(dim):
        alpha_k.append(alpha)
    
    
    
    while flag==1 and k<nmax:
        print('k:',k)
        
        x_k=X[k]
        value=functions(problem,x_k,theta)
        print('value=',value)
        d_k.append(compute_d(problem,x_k,theta))
        
        print('d_k:',d_k[k])
        
        if dim==1:
            if d_k[k]==0:
                flag=0
                break
        else:
            temp=0
            for i in range(len(d_k[k])):
                if d_k[k][i]!=0:
                    break
                else:
                    temp+=1
            if temp==len(d_k[k]):
                flag=0
                break
        temp=np.array(X[k])+alpha_k*np.array(d_k[k])[0]
        temp=list(temp)
        print('temp:',temp)
        X.append(temp)
        
        print('X:',X)
        k+=1
    #printTable(X)
    min_value=functions(problem,X[k],theta)
    print('min_value=',min_value)
    return X,d_k


# alpha choosen

# In[9]:


X,d_k=NewtonsMethod(3,x_3_1,0.5,10,10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




