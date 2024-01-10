from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications, optimizers
import numpy as np
from keras import backend as K
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

si=0
sid=0
sid_list=[]

full_path = data_path + "sampB3_a10.csv"

dat2 = pd.read_csv(full_path)
new_columns=dat2.columns.values
new_columns[0]="syB"
dat2.columns=new_columns
X = dat2.drop('syB', axis=1).values
Y = dat2[['syB']].values

kfold = KFold(n_splits=10, random_state=None, shuffle=True)
layers=[3,4,5]
nodes=[50,200,500,1000]
res_array = np.zeros( (10,3,4))
# enumerate splits
k=-1
for train, test in kfold.split(X):
    k=k+1
    print('train: %s, test: %s' % (train, test))
    i=-1
    for l_inx in layers:
        i=i+1
        j=-1
        for n_inx in nodes:
            j=j+1
            print(str(l_inx) + ", " + str(n_inx))
            l_num=2
            model = Sequential()
            model.add(Dense(n_inx, input_dim=6, activation='relu'))
            while l_num < l_inx:
                model.add(Dense(n_inx, activation='relu'))
                l_num = l_num + 1

            model.add(Dense(1, activation='linear'))
        # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
            learning_rate_list = [.1, .01, .001, .0001, .00001]
            xlr=learning_rate_list[2]
            xlr=0.001
            opt = optimizers.Adam(learning_rate = xlr)
            #opt = optimizers.SGD(learning_rate = xlr)
            model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
            fn_best_model='best_model' +'.h5'
            mc = ModelCheckpoint(fn_best_model, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            model.fit(
                X[train,],
                Y[train,],
                # batch_size = 32,
                validation_split = .2,
                epochs = 200,
                shuffle = True,
                callbacks = [es, mc],
                verbose = 2
                )
            saved_model = load_model(fn_best_model)
            
            Yp = saved_model.predict(X[test,])
            res_array[k,i,j]=np.sqrt(np.mean((Yp-Y[test,])**2))
        
resx=res_array.mean(axis=(0))
m_inx=np.where(resx == np.min(resx))        
l_inx=layers[int(m_inx[0])]
n_inx=nodes[int(m_inx[1])]
l_num=2
model = Sequential()
model.add(Dense(n_inx, input_dim=6, activation='relu'))
while l_num < l_inx:
    model.add(Dense(n_inx, activation='relu'))
    l_num = l_num + 1

model.add(Dense(1, activation='linear'))
opt = optimizers.Adam(learning_rate = xlr)    
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
fn_best_model='best_model' +'.h5'
mc = ModelCheckpoint(fn_best_model, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
model.fit(
        X,
        Y,
        # batch_size = 32,
        validation_split = .2,
        epochs = 200,
        shuffle = True,
        callbacks = [es, mc],
        verbose = 2
        )
saved_model = load_model(fn_best_model)
get_last_layer_output = K.function([saved_model.layers[0].input],
                              [saved_model.layers[len(saved_model.layers)-2].output])
layer_output = get_last_layer_output(tf.convert_to_tensor(X))[0] 
df = layer_output[:, (layer_output != 0).any(axis=0)]
if df.size==0:
    si=si+1

lm=LinearRegression()
lm_fit=lm.fit(df,Y)
Y_pred=lm_fit.predict(df)

# Extract Y hats from saved_model
full_A_path = data_path + "sampA3_a10.csv"
test_data = pd.read_csv(full_A_path)

Xt = test_data.values[:,1:7]
#Xt_n=normalize(Xt, axis=0, norm='l1')
Xt_n=Xt
am=np.mean(Xt, axis=0)
sd=np.std(Xt, axis=0, ddof=1)
#X_n[:,0]=(X[:,0]-am[0])/sd[0]
#Xt_n[:,1]=(Xt[:,1]-am[1])/sd[1]
#Xt_n[:,2]=(Xt[:,2]-am[2])/sd[2]
#Xt_n[:,3]=(Xt[:,3]-am[3])/sd[3]
#Xt_n[:,4]=(Xt[:,4]-am[4])/sd[4]
Wt = test_data.values[:,7]
Yp = saved_model.predict(Xt)
#Yp_unscaled = scaler.inverse_transform(Yp)
layer_output_A = get_last_layer_output(tf.convert_to_tensor(Xt))[0]
Yp2=lm_fit.predict(layer_output_A[:, (layer_output != 0).any(axis=0)])
#Yp2_unscaled = scaler.inverse_transform(Yp2)
est=Yp2.T@Wt/Wt.sum()
est2=Yp.T@Wt/Wt.sum()
gamai=Yp2-est

## Va
na=Yp.size
var_array_1=1/na*(1-na/N)*np.var(gamai, ddof=1) + 1/na/(N-1)*sum(map(lambda i : i * i, gamai))
## Vb
N_hat=Wt.sum()
H=np.append(np.ones([len(df),1]),df,1)
Hp = np.transpose(H) @ H
#np.linalg.inv(Hp)
if np.linalg.det(Hp)==0:
    sid=sid+1
    Hp_inv=np.linalg.pinv(Hp)
    sid_list.append(sid)
else:
    Hp_inv=np.linalg.inv(Hp)
H_A=np.append(np.ones([len(layer_output_A),1]),layer_output_A[:, (layer_output != 0).any(axis=0)],1)
tau=Hp_inv @ (np.transpose(H_A) @ Wt)
eta=np.multiply((Y - Y_pred), (H @ tau).reshape(Y.size,1))
eta_m=eta.sum()/N_hat

var_array_2=(np.square(eta-eta_m).sum()+eta_m*eta_m*(N-Y.size))/N_hat/(N_hat-1)

## V=Va+Vb
var_array=var_array_1+var_array_2

print(str(*est)+","+str(*var_array)+","+str(*est2))
