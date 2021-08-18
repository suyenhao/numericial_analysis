#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def functionValue(problem,x,theta):
    if problem==2:
        value=5*np.power(x[0][0],2)+5*np.power(x[1][0],2)-x[0][0]*x[1][0]-11*x[0][0]+11*x[1][0]+11
    elif problem==3:
        value=5*np.power(x[0][0],2)+5*np.power(x[1][0],2)-9*x[0][0]*x[1][0]+4*x[0][0]-15*x[1][0]+13
    elif problem==4:
        value=5*np.power(x[0][0],2)+20*np.power(x[1][0],2)+(3/2)*np.power(x[2][0],2)-18*x[0][0]*x[1][0]+2*x[0][0]*x[2][0]-x[1][0]*x[2][0]-12*x[0][0]-47*x[1][0]-8*x[2][0]
    elif problem==5:
        value=-9*x[0][0]-10*x[1][0]+theta*(-np.log(100-x[0][0]-x[1][0])-np.log(x[0][0])-np.log(x[1][0])-np.log(50-x[0][0]+x[1][0]))
    return value


# In[3]:


def createQq(problem):
    if problem==2:
        Q=np.array([
            [10,-1],
            [-1,10]
        ])
        q=np.array([
            [-11],
            [11]
        ])
    elif problem==3:
        Q=np.array([
            [10,-9],
            [-9,10]
        ])
        q=np.array([
            [4],
            [-15]
        ])
    elif problem==4:
        Q=np.array([
            [10,-18,2],
            [-18,40,-1],
            [2,-1,3]
        ])
        q=np.array([
            [12],
            [-47],
            [-8]
        ])
    return Q,q


# In[4]:


def height(problem,X,Y,theta):
    if problem==2:
        value=5*np.power(X,2)+5*np.power(Y,2)-X*Y-11*X+11*Y+11
    elif problem==3:
        value=5*np.power(X,2)+5*np.power(Y,2)-9*X*Y+4*X-15*Y+13
    elif problem==5:
        value=-9*X-10*Y+theta*(-np.log(100-X-Y)-np.log(X)-np.log(Y)-np.log(50-X+Y))
    return  value


# In[5]:


def height3D(problem,X,Y,Z,theta):
    if problem==4:
        value=5*np.power(X,2)+20*np.power(Y,2)+(3/2)*np.power(Z,2)-18*X*Y+2*X*Z-Y*Z-12*X-47*Y-8*Z
    return value


# In[6]:


def figure(problem,theta,n,X_list):
    if problem==2:
        x = np.linspace(-1,3,n)
        y = np.linspace(-4,2,n)
    elif problem==3:
        x = np.linspace(-1,12,n)
        y = np.linspace(-1,8,n) 
    elif problem==5:
        if theta==10:
            x = np.linspace(-1,40,n)
            y = np.linspace(10,100,n)
        elif theta==100:
            x = np.linspace(0,40,n)
            y = np.linspace(10,100,n)
    X,Y = np.meshgrid(x,y)
    # 填充顏色
    #plt.contourf(X,Y,height(problem,X,Y,theta),8,alpha=0.75,cmap=plt.cm.hot)
    # add contour lines
    C = plt.contour(X,Y,height(problem,X,Y,theta),8)
    # 顯示各等高線的資料標籤cmap=plt.cm.hot
    plt.clabel(C,inline=True,fontsize=10)
    x_list=[]
    y_list=[]
    for i in range(len(X_list)):
        plt.scatter(X_list[i][0][0], X_list[i][1][0], c='b', marker='x') 
        x_list.append(X_list[i][0][0])
        y_list.append(X_list[i][1][0])
    plt.plot(x_list,y_list,'--')
    plt.show()


# In[7]:


def figure3D(problem,theta,n):
    if problem==4:
        x = np.linspace(-1,3,n)
        y = np.linspace(-4,2,n)
        z = np.linspace(-4,2,n)
    X,Y,Z = np.meshgrid(x,y,z)
    # 填充顏色
    #plt.contourf(X,Y,height(problem,X,Y,theta),8,alpha=0.75,cmap=plt.cm.hot)
    # add contour lines
    C = plt.contour(X,Y,Z,height3D(problem,X,Y,Z,theta),8)
    # 顯示各等高線的資料標籤cmap=plt.cm.hot
    plt.clabel(C,inline=True,fontsize=10)
    plt.show()
    plt.savefig(problem+'level curve')


# In[8]:


def printTable(x,d,f_value):
    print('\\begin{center}')
    if problem==2 or problem ==3 or problem==5:
        print('\\begin{tabular}{| c | c | c | c | c | c | } ')
    elif problem==4:
        print('\\begin{tabular}{| c | c | c | c | c | c | c | c | } ')
    print('\\hline')
    if problem==2 or problem ==3 or problem==5:
        print('$k$ & $x_1$ & $x_2$ & $d_1$ & $d_2$ & $f(x)$ \\\\')
    elif problem==4:
        print('$k$ & $x_1$ & $x_2$ & $x_3$ & $d_1$ & $d_2$ & $d_3$ & $f(x)$ \\\\')
    print('\\hline\\hline')
    if problem ==3:
        for i in range(len(x)):
            print('%d & %f & %f & %f & %f & %f \\\\ '%(i,x[i][0][0],x[i][1][0],d[i][0][0],d[i][1][0],f_value[i]))
            print('\\hline')
    if problem==2 or problem ==5:
        for i in range(len(x)):
            print('%d & %f & %f & %f & %f & %f \\\\ '%(i,x[i][0][0],x[i][1][0],d[i][0][0],d[i][1][0],f_value[i]))
            print('\\hline')
    elif problem==4:
        for i in range(len(x)-1):
            print('%d & %f & %f & %f & %f & %f & %f & %f \\\\ '%(i,x[i][0][0],x[i][1][0],x[i][2][0],d[i][0][0],d[i][1][0],d[i][2][0],f_value[i]))
            print('\\hline')
    print('\\end{tabular}')
    
    print('\\end{center}')
    


# In[9]:


def compute_d(problem,x,theta):
    if problem==5:
        return -np.array([
            [-9+theta*(1/(100-x[0][0]-x[1][0])-1/x[0][0]+1/(50-x[0][0]+x[1][0]))],
            [-10+theta*(1/(100-x[0][0]-x[1][0])-1/x[1][0]-1/(50-x[0][0]+x[1][0]))]
        ])
    else:
        Q,q=createQq(problem)
        return -Q.dot(x)-q


# In[10]:


def computeAlpha(problem,x,d_k):
    if problem==5:
        alpha=10 #initial alpha
        nmax=50 #maximum step
        i=0
        #print('x:',x)
        f_value=functionValue(problem,x,theta) #initial function value
        while i<nmax:
            #print('i:',i)
            new_x=x+alpha*d_k
            new_f_value=functionValue(problem,new_x,theta)
            if new_x[0][0]>0 and new_x[1][0]>0 and new_x[0][0]+new_x[1][0]<100 and new_x[0][0]-new_x[1][0]<50 and new_f_value<f_value:
                alpha_bar=alpha
                #print('new_f_value:',new_f_value)
                break
            else:
                if i==nmax-1:
                    alpha_bar=alpha
                    break
                alpha=alpha/2                   
            i=i+1
        return alpha_bar
        
    else:
        Q,q=createQq(problem)
        d=compute_d(problem,x,theta)
        alpha_bar=d.T.dot(d)/(d.T.dot(Q.dot(d)))
        return alpha_bar


# In[11]:


def steepestDescent(problem,x_0,theta):
    k=0
    flag=1
    x=[]
    d=[]
    alpha=[]
    f_value=[]
    nmax=50
    n=256
    tol=1e-3
    x.append(x_0)
    #d_k=compute_d(problem,x_0,theta)
    #d.append(d_k)
    value=functionValue(problem,x[k],theta)
    f_value.append(value)
    
    while flag==1: #stop critian
        d_k=compute_d(problem,x[k],theta)
        d.append(d_k)
        if np.linalg.norm(d[k])==0:
            flag=0
            break
        
        alpha_k=computeAlpha(problem,x[k],d[k])
        alpha.append(alpha_k)
        
        new_x=x[k]+alpha[k]*d[k]
        x.append(new_x)
        
        value=functionValue(problem,x[k+1],theta)
        f_value.append(value)
        if problem==5:
            if abs((f_value[k-1]-f_value[k])/np.linalg.norm(x[k-1]-x[k]))<tol:
                flag=0
                break
        k+=1
        if k>nmax:
            flag=0
            break     
    d.append([[0],[0]])
    
    printTable(x,d,f_value)
    #for i in range(len(x)):
        #print('%d %f %f' %(i,d[i][0][0],d[i][1][0]))
    if problem==4:
        print('No figure!')
    else:
        figure(problem,theta,n,x)
    #figure3D(problem,theta,n)
    
    return x


# In[12]:


theta=0
problem=2
x2=[[0],[0]]
x=steepestDescent(problem,x2,theta)


# In[13]:


theta=0
problem=3
x3_1=[[0],[0]]
x=steepestDescent(problem,x3_1,theta)


# In[14]:


theta=0
problem=3
x3_2=[[-0.4],[0]]
x=steepestDescent(problem,x3_2,theta)


# In[15]:


theta=0
problem=3
x3_3=[[10],[0]]
x=steepestDescent(problem,x3_3,theta)


# In[16]:


theta=0
problem=3
x3_4=[[11],[0]]
x=steepestDescent(problem,x3_4,theta)


# In[17]:


#problem4
problem=4
x4_1=[[0],[0],[0]]
x=steepestDescent(problem,x4_1,theta)


# In[18]:


problem=4
x4_2=[[15.09],[7.66],[-6.56]]
x=steepestDescent(problem,x4_2,theta)


# In[19]:


problem=4
x4_3=[[11.77],[6.42],[-4.28]]
x=steepestDescent(problem,x4_3,theta)


# In[20]:


problem=4
x4_4=[[4.46],[2.25],[1.85]]
x=steepestDescent(problem,x4_4,theta)


# In[21]:


problem=5
x5_1=[[8],[90]]
theta=10
x=steepestDescent(problem,x5_1,theta)


# In[22]:


problem=5
x5_2=[[1],[40]]
theta=10
x=steepestDescent(problem,x5_2,theta)


# In[23]:


problem=5
x5_3=[[15],[68.69]]
theta=10
x=steepestDescent(problem,x5_3,theta)


# In[24]:


problem=5
x5_4=[[10],[20]]
theta=10
x=steepestDescent(problem,x5_4,theta)


# In[25]:


problem=5
x5_1=[[8],[90]]
theta=100
x=steepestDescent(problem,x5_1,theta)


# In[26]:


problem=5
x5_2=[[1],[40]]
theta=100
x=steepestDescent(problem,x5_2,theta)


# In[27]:


problem=5
x5_3=[[15],[68.69]]
theta=100
x=steepestDescent(problem,x5_3,theta)


# In[28]:


problem=5
x5_4=[[10],[20]]
theta=100
x=steepestDescent(problem,x5_4,theta)


# In[ ]:





# In[ ]:





# In[ ]:




