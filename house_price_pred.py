

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adam

df=pd.read_csv("dataset.csv")
print(df.head())

df=df.drop(['PRT_ID'],axis=1)

print(df.isnull().sum())
df=df.fillna(df.mean())
print(df.isnull().sum())

print(df['AREA'].unique())
df['AREA']=df['AREA'].replace(to_replace=["Karapakam"],value ="Karapakkam")
df['AREA']=df['AREA'].replace(to_replace=["KKNagar"],value ="KK Nagar")
df['AREA']=df['AREA'].replace(to_replace=["Ann Nagar","Ana Nagar"],value ="Anna Nagar")
df['AREA']=df['AREA'].replace(to_replace=["Adyr"],value ="Adyar")
df['AREA']=df['AREA'].replace(to_replace=["Velchery"],value ="Velachery")
df['AREA']=df['AREA'].replace(to_replace=["TNagar"],value ="T Nagar")
df['AREA']=df['AREA'].replace(to_replace=["Chrompt","Chrmpet","Chormpet"],value ="Chrompet")
print(df['AREA'].unique())

print(df['SALE_COND'].unique())
df['SALE_COND']=df['SALE_COND'].replace(to_replace=["Ab Normal"],value ="AbNormal")
df['SALE_COND']=df['SALE_COND'].replace(to_replace=["Adj Land"],value ="AdjLand")
df['SALE_COND']=df['SALE_COND'].replace(to_replace=["Partiall","PartiaLl"],value ="Partial")
print(df['SALE_COND'].unique())

print(df['UTILITY_AVAIL'].unique())
df['UTILITY_AVAIL']=df['UTILITY_AVAIL'].replace(to_replace=["NoSewr "],value ="NoSeWa")
df['UTILITY_AVAIL']=df['UTILITY_AVAIL'].replace(to_replace=["All Pub"],value ="AllPub")
print(df['UTILITY_AVAIL'].unique())

print(df['STREET'].unique())
df['STREET']=df['STREET'].replace(to_replace=["Pavd"],value ="Paved")
df['STREET']=df['STREET'].replace(to_replace=["No Access"],value ="NoAccess")
print(df['STREET'].unique())

print(df['BUILDTYPE'].unique())
df['BUILDTYPE']=df['BUILDTYPE'].replace(to_replace=["Others"],value ="Other")
df['BUILDTYPE']=df['BUILDTYPE'].replace(to_replace=["Comercial"],value ="Commercial")
print(df['BUILDTYPE'].unique())

print(df['PARK_FACIL'].unique())
df['PARK_FACIL']=df['PARK_FACIL'].replace(to_replace=["Noo"],value ="No")
print(df['PARK_FACIL'].unique())

def getmonth(date,df):
    l=[]
    month=[]
    for i in df.index:
       l=df[date][i].split('-')
       month.append(int(l[1]))
    return month

def getyear(date,df):
    l=[]
    year=[]
    for i in df.index:
       l=df[date][i].split('-')
       year.append(int(l[2]))
    return year

month=[]
year=[]
month=getmonth("DATE_SALE",df)
df.insert(2,"MONTH_SALE",month, True) 
year=getyear("DATE_SALE",df)
df.insert(3,"YEAR_SALE",year,True)
df=df.drop("DATE_SALE",axis=1)
print(df.head())

month=getmonth("DATE_BUILD",df)
df.insert(11,"MONTH_BUILD",month, True) 
year=getyear("DATE_BUILD",df)
df.insert(12,"YEAR_BUILD",year,True)
df=df.drop("DATE_BUILD",axis=1)
print(df.head())

def isfloat(df):
    try:
        float(df)
    except:
        return False
    return True

print(df[~df['INT_SQFT'].apply(isfloat)].head())

df['PARK_FACIL']=df['PARK_FACIL'].replace(to_replace=["No"],value =0)
df['PARK_FACIL']=df['PARK_FACIL'].replace(to_replace=["Yes"],value =1)

df['AREA']=df['AREA'].replace(to_replace=["Karapakkam"],value =0)
df['AREA']=df['AREA'].replace(to_replace=["KK Nagar"],value =1)
df['AREA']=df['AREA'].replace(to_replace=["Anna Nagar"],value =2)
df['AREA']=df['AREA'].replace(to_replace=["Adyar"],value =3)
df['AREA']=df['AREA'].replace(to_replace=["Velachery"],value =4)
df['AREA']=df['AREA'].replace(to_replace=["T Nagar"],value =5)
df['AREA']=df['AREA'].replace(to_replace=["Chrompet"],value =6)

df['SALE_COND']=df['SALE_COND'].replace(to_replace=["AbNormal"],value =0)
df['SALE_COND']=df['SALE_COND'].replace(to_replace=["AdjLand"],value =1)
df['SALE_COND']=df['SALE_COND'].replace(to_replace=["Partial"],value =2) 
df['SALE_COND']=df['SALE_COND'].replace(to_replace=["Family"],value =3)   
df['SALE_COND']=df['SALE_COND'].replace(to_replace=["Normal Sale"],value =4)    

df['STREET']=df['STREET'].replace(to_replace=["Paved"],value =0)
df['STREET']=df['STREET'].replace(to_replace=["NoAccess"],value =1)
df['STREET']=df['STREET'].replace(to_replace=["Gravel"],value =2)

df['UTILITY_AVAIL']=df['UTILITY_AVAIL'].replace(to_replace=["NoSeWa"],value =0)
df['UTILITY_AVAIL']=df['UTILITY_AVAIL'].replace(to_replace=["AllPub"],value =1)
df['UTILITY_AVAIL']=df['UTILITY_AVAIL'].replace(to_replace=["ELO"],value =2)

df['BUILDTYPE']=df['BUILDTYPE'].replace(to_replace=["Other"],value =0)
df['BUILDTYPE']=df['BUILDTYPE'].replace(to_replace=["Commercial"],value =1)
df['BUILDTYPE']=df['BUILDTYPE'].replace(to_replace=["House"],value =2)

df['MZZONE']=df['MZZONE'].replace(to_replace=['RH'],value =0)
df['MZZONE']=df['MZZONE'].replace(to_replace=['RM'],value =1)
df['MZZONE']=df['MZZONE'].replace(to_replace=['I'],value =2)
df['MZZONE']=df['MZZONE'].replace(to_replace=['RL'],value =3)
df['MZZONE']=df['MZZONE'].replace(to_replace=['A'],value =4)
df['MZZONE']=df['MZZONE'].replace(to_replace=['C'],value =5)

variables=df[['AREA', 'INT_SQFT', 'MONTH_SALE', 'YEAR_SALE', 'DIST_MAINROAD', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM', 'SALE_COND', 'PARK_FACIL', 'MONTH_BUILD', 'YEAR_BUILD', 'BUILDTYPE', 'UTILITY_AVAIL', 'STREET', 'MZZONE', 'QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL', 'REG_FEE', 'COMMIS']]
vif=pd.DataFrame()
df1=add_constant(variables)
vif["VIF"]=[variance_inflation_factor(df1.values,i) for i in range(df1.shape[1])]
vif["features"]=df.columns
print(vif)

df=df.drop("QS_ROOMS",axis=1)
df=df.drop("QS_BATHROOM",axis=1)
df=df.drop("QS_BEDROOM",axis=1)
df=df.drop("QS_OVERALL",axis=1)

Y=df['SALES_PRICE'].values
X=df.drop('SALES_PRICE',axis=1).values

X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=0.33, random_state=101)

s_scaler=StandardScaler()
X_train=s_scaler.fit_transform(X_train.astype(np.float))
X_test=s_scaler.transform(X_test.astype(np.float))


print(X_train.shape)
print("\n\n--------------Training Model 1-----------------")

model1=Sequential()
model1.add(Dense(45,activation='relu',input_shape=(18,)))
model1.add(Dense(30,activation='relu'))
model1.add(Dense(15,activation='relu'))
model1.add(Dense(1))
opti=Adam(learning_rate=0.01)
model1.compile(optimizer=opti,loss='mean_squared_error')

history1=model1.fit(x=X_train,y=Y_train,validation_data=(X_test,Y_test),batch_size=128,epochs=300,verbose=1)
Y_pred1=model1.predict(X_test)

print("\n\n--------------Training Model 2-----------------")

model2=Sequential()
model2.add(Dense(80,activation='tanh',input_shape=(18,)))
model2.add(Dropout(0.2))
model2.add(Dense(40,activation='relu'))
model2.add(Dropout(0.1))
model2.add(Dense(20,activation='relu'))
model2.add(Dense(10,activation='relu'))
model2.add(Dense(1))
opti=Adadelta(learning_rate=0.03)
model2.compile(optimizer=opti,loss='mean_squared_error')

history2=model2.fit(x=X_train,y=Y_train,validation_data=(X_test,Y_test),batch_size=128,epochs=100,verbose=1)
Y_pred2=model2.predict(X_test)


print("\n\n--------------Model 1-----------------")

print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,Y_pred1))  
print('Mean Squared Error:',metrics.mean_squared_error(Y_test,Y_pred1))  
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(Y_test,Y_pred1)))
print('Variance Score(in %):',metrics.explained_variance_score(Y_test,Y_pred1)*100)
print('R2 Score(in %):',metrics.r2_score(Y_test,Y_pred1)*100)
print('Plot for model 1:')

loss_train1 = history1.history['loss']
loss_val1 = history1.history['val_loss']
epochs = range(1,301)
plt.plot(epochs, loss_train1, 'r', label='Training loss')
plt.title('Training loss for model 1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("\n\n--------------Model 2-----------------")

print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,Y_pred2))  
print('Mean Squared Error:',metrics.mean_squared_error(Y_test,Y_pred2))  
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(Y_test,Y_pred2)))
print('Variance Score(in %):',metrics.explained_variance_score(Y_test,Y_pred2)*100)
print('R2 Score(in %):',metrics.r2_score(Y_test,Y_pred2)*100)
print('Plot for model 2:')

loss_train2 = history2.history['loss']
loss_val2 = history2.history['val_loss']
epochs = range(1,101)
plt.plot(epochs, loss_train2, 'r', label='Training loss')
plt.title('Training loss for model 2')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

def pred_price(AREA,INT_SQFT,MONTH_SALE,YEAR_SALE,DIST,BEDROOM,BATHROOM,ROOM,SALE_COND,PARK,MONTH_BUILD,YEAR_BUILD,BUILD,UTILITY,STREET,MZZONE,REG_FEE,COMMIS):
    X1=np.zeros(len(18))
    X1[0]=AREA
    X1[1]=INT_SQFT
    X1[2]=MONTH_SALE
    X1[3]=YEAR_SALE
    X1[4]=DIST
    X1[5]=BEDROOM
    X1[6]=BATHROOM
    X1[7]=ROOM
    X1[8]=SALE_COND
    if(PARK=="YES"):
        X1[9]=1
    else:
        X1[9]=0
    X1[10]=MONTH_BUILD
    X1[11]=YEAR_BUILD
    X1[12]=BUILD
    X1[13]=UTILITY
    X1[14]=STREET
    X1[15]=MZZONE
    X1[16]=REG_FEE
    X1[17]=COMMIS
    X1[0]=X1[0].replace(to_replace=["Karapakkam"],value =0)
    X1[0]=X1[0].replace(to_replace=["KK Nagar"],value =1)
    X1[0]=X1[0].replace(to_replace=["Anna Nagar"],value =2)
    X1[0]=X1[0].replace(to_replace=["Adyar"],value =3)
    X1[0]=X1[0].replace(to_replace=["Velachery"],value =4)
    X1[0]=X1[0].replace(to_replace=["T Nagar"],value =5)
    X1[0]=X1[0].replace(to_replace=["Chrompet"],value =6)
    
    X1[8]=X1[8].replace(to_replace=["AbNormal"],value =0)
    X1[8]=X1[8].replace(to_replace=["AdjLand"],value =1)
    X1[8]=X1[8].replace(to_replace=["Partial"],value =2) 
    X1[8]=X1[8].replace(to_replace=["Family"],value =3)   
    X1[8]=X1[8].replace(to_replace=["Normal Sale"],value =4) 
    
    X1[14]=X1[14].replace(to_replace=["Paved"],value =0)
    X1[14]=X1[14].replace(to_replace=["NoAccess"],value =1)
    X1[14]=X1[14].replace(to_replace=["Gravel"],value =2)
    
    X1[12]=X1[12].replace(to_replace=["Other"],value =0)
    X1[12]=X1[12].replace(to_replace=["Commercial"],value =1)
    X1[12]=X1[12].replace(to_replace=["House"],value =2)
    
    X1[15]=X1[15].replace(to_replace=['RH'],value =0)
    X1[15]=X1[15].replace(to_replace=['RM'],value =1)
    X1[15]=X1[15].replace(to_replace=['I'],value =2)
    X1[15]=X1[15].replace(to_replace=['RL'],value =3)
    X1[15]=X1[15].replace(to_replace=['A'],value =4)
    X1[15]=X1[15].replace(to_replace=['C'],value =5)
    
    X1=s_scaler.transform(X1.astype(np.float))
    
    return str(model1.predict([X1])[0])

AREA=input()
INT_SQFT=float(input())
MONTH_SALE=int(input())
YEAR_SALE=int(input())
DIST=float(input())
BEDROOM=int(input())
BATHROOM=int(input())
ROOM=int(input())
SALE_COND=input()
PARK=input()
MONTH_BUILD=int(input())
YEAR_BUILD=int(input())
BUILD=input()
UTILITY=input()
STREET=input()
MZZONE=input()
REG_FEE=float(input())
COMMIS=float(input())

print(pred_price(AREA,INT_SQFT,MONTH_SALE,YEAR_SALE,DIST,BEDROOM,BATHROOM,ROOM,SALE_COND,PARK,MONTH_BUILD,YEAR_BUILD,BUILD,UTILITY,STREET,MZZONE,REG_FEE,COMMIS))
