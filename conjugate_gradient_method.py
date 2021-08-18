#!/usr/bin/env python
# coding: utf-8

# conjugate gradient method ex4 

# In[60]:


import numpy as np


# In[61]:


def createQq(problem):
    if problem==4:
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


# In[62]:


def printTable(problem,x,d):
    if problem==4:
        print('\\begin{center}')
        print('\\begin{tabular}{|| c c ||} ')
        print('\\hline')
        print('k & $x_1$ & $x_2$ & $x_3$ & $d_1$ & $d_2$ & $d_3$\\\\')
        print('\\hline\\hline')
        for i in range(1,len(x)):
            print('%d & %f & %f & %f & %f & %f & %f  \\\\'%(i,x[i][0][0],x[i][1][0],x[i][2][0],d[i][0][0],d[i][1][0],d[i][2][0]))
            print('\\hline')
        print('\\end{tabular}')
        print('\\end{center}')


# In[63]:


def creatx(problem,number):
    if problem==4:
        if number==1:
            x=[0]
            x.append([[0],[0],[0]])
        elif number==2:
            x=[0]
            x.append([[15.09],[7.66],[-6.56]])
        elif number==3:
            x=[0]
            x.append([[11.77],[6.42],[-4.28]])
        elif number==4:
            x=[0]
            x.append([[4.46],[2.25],[1.85]])
    return x


# In[64]:


def cgmethod(problem,number):
    Q,q=createQq(problem)
    x=creatx(problem,number)
    k=1
    r=[[0]]
    r.append(-(Q.dot(x[1])+q))
    d=[[0]]
    d.append(r[k])
    t=[[0]]
    s=[[0]]
    Max=20
    TOL=1e-3
    leaveloop=1
    while leaveloop==1:
        t.append(np.dot(r[k].T,d[k])/d[k].T.dot(Q.dot(d[k])))
        x_k=x[k]+t[k]*d[k]
        x.append(x_k)
        r.append(r[k]-t[k]*Q.dot(d[k]))
        s.append(d[k].T.dot(Q.dot(r[k+1]))/d[k].T.dot(Q.dot(d[k])))
        d.append(r[k+1]+s[k]*d[k])
        if k>=Max or np.linalg.norm(r[k])<=TOL:
            leaveloop=0
        k=k+1    
    #for i in range(1,len(x)):
       # print("x_%d =" % (i))
        #print(x[i])
    printTable(problem,x,d)
    return x


# In[65]:


problem=4
number=1
x=cgmethod(problem,number)


# In[66]:


problem=4
number=2
x=cgmethod(problem,number)


# In[67]:


problem=4
number=3
x=cgmethod(problem,number)


# In[68]:


problem=4
number=4
x=cgmethod(problem,number)


# conjugate gradient method Ex5

# In[43]:


import matplotlib.pyplot as plt
import numpy as np


# In[44]:


def gradientF(problem,x,theta):
    if problem==5:
        temp=np.array([
                [-9+theta*(1/(100-x[0][0]-x[1][0])-1/x[0][0]+1/(50-x[0][0]+x[1][0]))],
                [-10+theta*(1/(100-x[0][0]-x[1][0])-1/x[1][0]-1/(50-x[0][0]+x[1][0]))]
            ])
    return temp


# In[45]:


def hessianF(problem,x,theta):
    if problem==5:
        return np.array([
            [theta*(1/np.power(100-x[0][0]-x[1][0],2)+1/np.power(x[0][0],2)+1/np.power(50-x[0][0]+x[1][0],2)), theta*(1/np.power(100-x[0][0]-x[1][0],2)-1/np.power(50-x[0][0]+x[1][0],2))],
            [theta*(1/np.power(100-x[0][0]-x[1][0],2)-1/np.power(50-x[0][0]+x[1][0],2)), theta*(1/np.power(100-x[0][0]-x[1][0],2)+1/np.power(x[1][0],2)+1/np.power(50-x[0][0]+x[1][0],2))]
        ])


# In[46]:


def Isdomain(x,d,t):
    flag=1
    while flag==1:
        if x[0][0]+t*d[0][0]>0 and x[1][0]+t*d[1][0]>0 and x[0][0]+t*d[0][0]+x[1][0]+t*d[1][0]<100 and  x[0][0]+t*d[0][0]-x[1][0]-t*d[1][0]<50:
            flag=0
            return t
        t=t/2


# In[47]:


def creatx(problem,number):    
    if problem==5:
        if number==1:
            x=[[8],[90]]
        elif number==2:
            x=[[1],[40]]
        elif number==3:
            x=[[15],[68.69]]
        elif number==4:
            x=[[10],[20]]
    return x


# In[48]:


def printx(problem,x,d):
    if problem==5:
        print('\\begin{center}')
        print('\\begin{tabular}{|| c c ||} ')
        print('\\hline')
        print('k & $x_1$ & $x_2$ & $d_1$ & $d_2$  \\\\')
        print('\\hline\\hline')
        for i in range(len(x)):
            print('%d & %f & %f & %f & %f     \\\\'%(i,x[i][0][0],x[i][1][0],d[i][0],d[i][1]))
            print('\\hline')
        print('\\end{tabular}')
        print('\\end{center}')


# In[49]:


def height(problem,X,Y,theta):
    if problem==5:
        value=-9*X-10*Y+theta*(-np.log(100-X-Y)-np.log(X)-np.log(Y)-np.log(50-X+Y))
    return  value


# In[50]:


def figure(problem,theta,n,X_list):
    if problem==5:
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


# In[51]:


def CGmethod(problem,number,theta):
    n=256
    Max=20
    TOL=1e-3
    x=[]
    r=[]
    d=[]
    t=[]
    s=[]
    k=0
    x.append(np.array(creatx(problem,number)))
    q=gradientF(problem,x[k],theta)
    r.append(q)
    d.append(-r[k])
    leaveloop=1
    while leaveloop==1:   
        Q=hessianF(problem,x[k],theta)
        d_Q_d=d[k].T.dot(Q.dot(d[k]))
        t_k=-np.dot(r[k].T,d[k])/d_Q_d
        #t.append(-np.dot(r[k].T,d[k])/d[k].T.dot(Qk[k].dot(d[k])))
        t_k=Isdomain(x[k],d[k],t_k)
        t.append(t_k)
        x_k=x[k]+t[k]*d[k]
        x.append(x_k)
        q_k=gradientF(problem,x[k+1],theta)
        r.append(q_k)
        s.append((r[k+1].T.dot(Q.dot(d[k])))/d_Q_d)
        d.append(-r[k+1]+s[k]*d[k])
        if k>=Max or np.linalg.norm(r[k])<=TOL:
            leaveloop=0
        k=k+1
        
    printx(problem,x,d)
    figure(problem,theta,n,x)
    return x,d


# In[52]:


problem=5
number=1
theta=10
x,d=CGmethod(problem,number,theta)


# In[53]:


problem=5
number=1
theta=100
x=CGmethod(problem,number,theta)


# In[54]:


problem=5
number=2
theta=10
x=CGmethod(problem,number,theta)


# In[55]:


problem=5
number=2
theta=100
x=CGmethod(problem,number,theta)


# In[56]:


problem=5
number=3
theta=10
x=CGmethod(problem,number,theta)


# In[57]:


problem=5
number=3
theta=100
x=CGmethod(problem,number,theta)


# In[58]:


problem=5
number=4
theta=10
x=CGmethod(problem,number,theta)


# In[59]:


problem=5
number=4
theta=100
x=CGmethod(problem,number,theta)


# In[ ]:





# In[ ]:




