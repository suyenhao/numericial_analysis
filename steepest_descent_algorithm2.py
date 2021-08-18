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
        value=5*np.power(x[0][0],2)+20*np.power(x[1][0],2)+(3/2)*np.power(x[2][0],2)-18*x[0][0]*x[1][0]+2*x[0][0]*x[2][0]-x[1][0]*x[2][0]+12*x[0][0]-47*x[1][0]-8*x[2][0]
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


def compute_d(problem,x,theta):
    if problem==5:
        return -np.array([
            [-9+theta*(1/(100-x[0][0]-x[1][0])-1/x[0][0]+1/(50-x[0][0]+x[1][0]))],
            [-10+theta*(1/(100-x[0][0]-x[1][0])-1/x[1][0]-1/(50-x[0][0]+x[1][0]))]
        ])
    else:
        Q,q=createQq(problem)
        #print('Q:',Q)
        #print('q:',q)
        #print('-Q.dot(x)-q:',-Q.dot(x)-q)
        #print('x:', x)
        return -Q.dot(x)-q


# In[5]:


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


# In[6]:


def heigh(problem,x,y,theta):
    if problem==2:
        value=5*np.power(x,2)+5*np.power(y,2)-x*y-11*x+11*y+11
    elif problem==3:
        value=5*np.power(x,2)+5*np.power(y,2)-9*x*y+4*x-15*y+13
    elif problem==5:
        value=-9*x-10*y+theta*(-np.log(100-x-y)-np.log(x)-np.log(y)-np.log(50-x+y))
    return value


# In[7]:


def showfigure(problem,theta,n,X_list):
    if problem==2:
        #centerX=1.22222222
        #centerY=-1.22222222
        
        x = np.linspace(-4,7,n)
        y = np.linspace(-7,4,n)
        
        
    elif problem==3:
        #centerX=5
        #centerY=6
        x = np.linspace(-1,13,n)
        y = np.linspace(-4,17,n)
        
    elif problem==5:
        if theta==10:
            x = np.linspace(-1,40,n)
            y = np.linspace(10,100,n)
        elif theta==100:
            x = np.linspace(0,40,n)
            y = np.linspace(10,100,n)
        
        
    # 將原始資料變成網格資料
    X,Y = np.meshgrid(x,y)
    # 填充顏色
    #plt.contourf(X,Y,height(X,Y),8,alpha=0.75,cmap=plt.cm.hot)

    # add contour lines 
    C = plt.contour(X,Y,heigh(problem,X,Y,theta),8)
    # 顯示各等高線的資料標籤cmap=plt.cm.hot
    plt.clabel(C,inline=True,fontsize=10)
    x_list=[]
    y_list=[]
    for i in range(len(X_list)):
        plt.scatter(X_list[i][0][0], X_list[i][1][0], c='b', marker='x') 
        x_list.append(X_list[i][0][0])
        y_list.append(X_list[i][1][0])
    plt.plot(x_list,y_list,'--')
    plt.title('Problem'+str(problem)+' with initial point: '+str(X_list[0]))
    #plt.savefig('Problem'+str(problem)+' 3-2')
    #plt.savefig('Problem'+str(problem)+' with initial point:'+str(X_list[0]))
    plt.show()


# In[8]:


#print table
def printTable(problem,x,d,alpha,f_value):
    print('\\begin{center}')
    if problem==2 or problem ==3 or problem==5:
        print('\\begin{tabular}{| c || c | c || c | c || c | c |} ')
    elif problem==4:
        print('\\begin{tabular}{| c || c | c | c || c | c | c || c | c |} ')
    print('\\hline')
    if problem==2 or problem ==3 or problem==5:
        print('$k$ & $x_1^k$ & $x_2^k$ & $d_1^k$ & $d_2^k$ & $\\alpha^k$ & $f(x^k)$ \\\\')
    elif problem==4:
        print('$k$ & $x_1^k$ & $x_2^k$ & $x_3^k$ & $d_1^k$ & $d_2^k$ & $d_3^k$ & $\\alpha^k$ & $f(x^k)$ \\\\')
    print('\\hline\\hline')
    if problem ==5:
        for i in range(len(x)):
            print('%d & %f & %f & %f & %f & %f & %f\\\\ '%(i,x[i][0][0],x[i][1][0],d[i][0][0],d[i][1][0],alpha[i],f_value[i]))
            print('\\hline')
    if problem==2 or problem ==3:
        for i in range(len(x)):
            print('%d & %f & %f & %f & %f & %f & %f\\\\ '%(i,x[i][0][0],x[i][1][0],d[i][0][0],d[i][1][0],alpha[i][0][0],f_value[i]))
            print('\\hline')
    elif problem==4:
        for i in range(len(x)-1):
            print('%d & %f & %f & %f & %f & %f & %f & %f & %f\\\\ '%(i,x[i][0][0],x[i][1][0],x[i][2][0],d[i][0][0],d[i][1][0],d[i][2][0],alpha[i][0][0],f_value[i]))
            print('\\hline')
    print('\\end{tabular}')
    
    print('\\end{center}')
    


# In[9]:


def steepestDescent(problem,x_0,theta):
    k=0
    flag=1
    x=[]
    d=[]
    alpha=[]
    f_value=[]
    nmax=100
    n=256 #level curve
    tolorance=1e-8
    
    x.append(x_0)
    #d_k=compute_d(problem,x[k],theta)
    #d.append(d_k)
    value=functionValue(problem,x[k],theta)
    f_value.append(value)
    
    while flag==1: #stop critian
        #print('xk:',x[k])
        d_k=compute_d(problem,x[k],theta)
        d.append(d_k)
        
        
        if np.linalg.norm(d[k])==0:
            flag=0
            break
        #print('dk:',d_k)
        #temp=0
        #for i in range(len(d[k])):
        #    if d[k][i]==0:
        #        temp+=1
        #if temp==len(d[k]):
        #    flag=0
        #    break
        
        alpha_k=computeAlpha(problem,x[k],d[k])
        alpha.append(alpha_k)
        
        new_x=x[k]+alpha[k]*d[k]
        x.append(new_x)
        
        value=functionValue(problem,x[k+1],theta)
        f_value.append(value)
        if problem==5:
            if abs((f_value[k-1]-f_value[k])/np.linalg.norm(x[k-1]-x[k]))<tolorance:
                flag=0
                break
        
        k+=1
        if k>nmax:
            flag=0
            break
    
    
    if problem==2 or problem==3 or problem==5:
        showfigure(problem,theta,n,x)
    #printTable(problem,x,d,alpha,f_value)
    print('x:',x)
    print('d:',d)
    #print('alpha:',alpha)
    #print('f_value:',f_value)
    #print('k:',k)
    print('The best solution:',x[-1])
    print('The minmum value:',f_value[-1])
    
    return x,d,alpha,f_value


# In[10]:


#problem2 (d)
#initial point (0,0)^T tolorance 10^-11
theta=0
problem=2
x2=[[0],[0]]
x,d,alpha,f_value=steepestDescent(problem,x2,theta)
f_value


# In[11]:


#problem3 
#initial point 
theta=0
problem=3
x3_1=[[0],[0]]
x,d,alpha,f_value=steepestDescent(problem,x3_1,theta)
f_value[1]


# In[12]:


theta=0
problem=3
x3_2=[[-0.4],[0]]
x,d,alpha,f_value=steepestDescent(problem,x3_2,theta)


# In[13]:


theta=0
problem=3
x3_3=[[10],[0]]
x,d,alpha,f_value=steepestDescent(problem,x3_3,theta)


# In[34]:


theta=0
problem=3
x3_4=[[11],[0]]
x,d,alpha,f_value=steepestDescent(problem,x3_4,theta)
f_value


# In[35]:


#problem4
theta=0
problem=4
x4_1=[[0],[0],[0]]
x,d,alpha,f_value=steepestDescent(problem,x4_1,theta)


# In[16]:


problem=4
theta=0
x4_2=[[15.09],[7.66],[-6.56]]
x,d,alpha,f_value=steepestDescent(problem,x4_2,theta)


# In[17]:


problem=4
theta=0
x4_3=[[11.77],[6.42],[-4.28]]
x,d,alpha,f_value=steepestDescent(problem,x4_3,theta)


# In[18]:


problem=4
theta=0
x4_4=[[4.46],[2.25],[1.85]]
x,d,alpha,f_value=steepestDescent(problem,x4_4,theta)


# problem 5 with $\theta=10$.

# In[19]:


x5_1=[[8],[90]]
x5_2=[[1],[40]]
x5_3=[[15],[68.69]]
x5_4=[[10],[20]]


# In[20]:


problem=5
theta=10
x,d,alpha,f_value=steepestDescent(problem,x5_1,theta)


# In[21]:


problem=5
theta=10
x,d,alpha,f_value=steepestDescent(problem,x5_2,theta)


# In[ ]:


problem=5
theta=10
x,d,alpha,f_value=steepestDescent(problem,x5_3,theta)


# In[ ]:


problem=5
theta=10
x,d,alpha,f_value=steepestDescent(problem,x5_4,theta)


# In[ ]:





# problem 5 with $\theta=100$.

# In[ ]:


problem=5
theta=100
x,d,alpha,f_value=steepestDescent(problem,x5_1,theta)


# In[ ]:


problem=5
theta=100
x,d,alpha,f_value=steepestDescent(problem,x5_2,theta)


# In[ ]:


problem=5
theta=100
x,d,alpha,f_value=steepestDescent(problem,x5_3,theta)


# In[ ]:


problem=5
theta=100
x,d,alpha,f_value=steepestDescent(problem,x5_4,theta)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




