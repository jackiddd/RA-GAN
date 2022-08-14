# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 20:20:36 2021

@author: win10
"""
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error 
from sklearn import datasets
from sklearn import svm
from sklearn import neighbors
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn import tree 
from sklearn import ensemble
from sklearn.neural_network import MLPRegressor  


'''============ Select hyperparameters ==================='''

filename='data'
dataset= 'CO2'    # 'diabete'
method='RAGAN/'  

num_train = 100
num_test = 200
num_generated_data = 1000

epoch =12000
N=10 
'''================== Import data ==========================='''


if dataset == 'CO2':
    x=np.load(filename+'/'+dataset+'/x.npy')
    y=np.load(filename+'/'+dataset+'/y.npy')
    
    data_real=np.concatenate((x,y.reshape(-1,1)),1)
    array = np.arange(0, len(x))
    np.random.seed(2)
    np.random.shuffle(array)
    data_real=data_real[array]
    
    data_trian=data_real[:num_train,:]
    data_test=data_real[-num_test:,:]
    data_gen=np.load('data/'+method+dataset+'_data_%d.npy'%epoch)[:num_generated_data,:]


if dataset == 'diabete':
    data=datasets.load_diabetes()
    data_X=data.data
    data_Y=data.target
    
    #打乱数据
    array = np.arange(0, len(data_X))
    np.random.seed(2)
    np.random.shuffle(array)    
    x_less=data_X[array]
    y_less=data_Y[array]
    
    data_all = np.hstack((x_less,y_less.reshape(-1,1)))
    #标准化
    data_all=(data_all-data_all.min(0))/(data_all.max(0)-data_all.min(0))

    data_trian=data_all[:num_train,:]
    data_test=data_all[-num_test:,:]    
    
    data_gen=np.load('data/'+method+dataset+'_data_%d.npy'%epoch)[:num_generated_data,:]

'''=============== Regression modeling (SVR) ==================='''
model = svm.SVR()
data_DA=np.vstack([data_trian,data_gen])

model.fit(data_DA[:,:-1],data_DA[:,-1])

'''================ Output prediction results ======================'''

pre=model.predict(data_test[:,:-1])
RMSE=np.sqrt(mean_squared_error(data_test[:,-1],pre))
R2=r2_score(data_test[:,-1],pre)
print('RMSE and R2 are ',RMSE,R2)

model_1 = svm.SVR()
model_2 = neighbors.KNeighborsRegressor(n_neighbors=15)
model_3 = tree.DecisionTreeRegressor()
model_4 = ensemble.RandomForestRegressor(n_estimators=100)
model_5 = MLPRegressor(solver='adam', alpha=1e-3, hidden_layer_sizes=(64,32,16), random_state=47,activation='relu', max_iter=5000)

def EICRS(x,y):
  eicrs=0
  ICRS=[]
  for j in [model_1,model_2,model_3,model_4,model_5]:
      
      model = j
      Rmse=0  
      for i in range(N):    
        num=int( (y.shape[0])/N )
        index= list(range(i*num,i*num+num))
        
        x_train = np.delete(x,index,0)  
        x_test =  x[index] 
        y_train = np.delete(y,index,0)  
        y_test =  y[index] 
        
        model.fit(x_train,y_train) 
        pre=model.predict(x_test)
        rmse=np.sqrt(mean_squared_error(y_test,pre))        
        Rmse += rmse    
      Rmse=Rmse/N
      
      eicrs+=Rmse
      ICRS.append(Rmse)
      
  ICRS =np.array(ICRS)   
  eicrs=eicrs/5
  return eicrs,ICRS

def EECRS(x_r, y_r, x_g, y_g):
  eecrs=0
  ECRS=[]
  for j in [model_1,model_2,model_3,model_4,model_5]:
    # for j in [model_1,model_2,model_4]:  

      model = j
      Rmse=0  

      for i in range(N):    
        num=int((y_r.shape[0])/N )
        index= list(range(i*num,i*num+num))
        x_train = np.delete(x_r,index,0)  
        x_test =  x_g[index] 
        y_train = np.delete(y_r,index,0)  
        y_test =  y_g[index] 
        
        model.fit(x_train,y_train) 
        pre=model.predict(x_test)
        rmse=np.sqrt(mean_squared_error(y_test,pre))   
        # print('真预测假，模型是{}，误差是{}'.format(model, rmse))

        Rmse += rmse   
      for i in range(N):    
        num_r=int( (y_r.shape[0])/N )
        num_g=int( (y_g.shape[0])/N )
        index_r= list(range(i*num_r,i*num_r+num_r))
        index_g= list(range(i*num_g,i*num_g+num_g))
        
        x_train = np.delete(x_g,index_g,0)  


        x_test =  x_r[index_r] 
        y_train = np.delete(y_g,index_g,0)  
        y_test =  y_r[index_r] 
        
        model.fit(x_train,y_train) 
        pre=model.predict(x_test)
        rmse=np.sqrt(mean_squared_error(y_test,pre))  
        
        # print('假预测真，模型是{}，误差是{}'.format(model, rmse))

        Rmse += rmse        
        
      Rmse=Rmse/(2*N)
      ECRS.append(Rmse)
      eecrs+=Rmse
      
  eecrs=eecrs/5
  ECRS=np.array(ECRS)
  return eecrs,ECRS

eicrs,ICRS = EICRS(data_gen[:,:-1],data_gen[:,-1]) 
print('EICRS={},They are '.format(eicrs),ICRS)
eecrs,ECRS= EECRS(data_trian[:,:-1],data_trian[:,-1],data_gen[:,:-1],data_gen[:,-1])
print('EECRS={},They are '.format(eecrs),ECRS)


