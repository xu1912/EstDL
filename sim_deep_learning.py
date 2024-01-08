from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications, optimizers
import numpy as np
from keras import backend as K
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from scipy import stats

data_path = "C:/Users/cxu2/Documents/EstDL/dat2/"
ns=1000
N=10000
bias_array = np.zeros( (ns, 2) )
est_array = np.zeros( (ns, 2) )
var_array = np.zeros( (ns, 3) )
true_mean=12.19999            #4n: -11.69378; 4e: -1.483135; 3n: 0.9452195; 3e: 2.000757; 1n: 0.9997838; 1e: 0.9994008; 
#1n-2k: 0.9999008; 2n-2k: 12.19999; 4n-2k: -11.69195

#true_median=0.9990163          #4n: -0.6511648; 4e: -1.395075; 3n: 1.619938; 3e: 1.439664; 1n: 0.9990163; 1e: 0.9448088;
fn_res = data_path + "res_3L_50n_lr01_bias_4l.txt"
fn_res2 = data_path + "res_3L_50n_lr01_est_4l.txt"
fn_res3 = data_path + "res_3L_50n_lr01_var_4l.txt"


i = 1
si=0
sid=0
sid_list=[]

for j in range(ns):
    i = j + 1
    full_path = data_path + "dtB_" + str(i) + ".csv"
    dat2 = pd.read_csv(full_path)
    # Data needs to be scaled to a small range like 0 to 1 for the neural
    # network to work well.

    new_columns=dat2.columns.values
    new_columns[0]="syB"
    dat2.columns=new_columns
    X = dat2.drop('syB', axis=1).values
    Y = dat2[['syB']].values

    # Define the model
    # ADD / SUBTRACT number of layers (try 2, 3, 4 output included)
    model = Sequential()
    model.add(Dense(150, input_dim=4, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # Change learning rate each time (range: .1, .01, .001, .0001, .00001)
    learning_rate_list = [.1, .01, .001, .0001, .00001]
    xlr=learning_rate_list[2]
    xlr=0.01
    opt = optimizers.Adam(learning_rate = xlr)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    fn_best_model='best_model' +'.h5'
    mc = ModelCheckpoint(fn_best_model, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    model.fit(
            X,
            Y,
            #batch_size = 64,
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
    li=(layer_output != 0).any(axis=0)
    lii=[i for i, x in enumerate(li) if x]

    if df.size==0:
        si=si+1
        continue
    lm=LinearRegression()
    lm_fit=lm.fit(df,Y)
    Y_pred=lm_fit.predict(df)
    
    # Extract Y hats from saved_model
    full_A_path = data_path + "dtA_" + str(i) + ".csv"
    full_Aw_path = data_path + "dtAw_" + str(i) + ".csv"
    test_data = pd.read_csv(full_A_path)
    test_data_w = pd.read_csv(full_Aw_path)
    Xt = test_data.values[:,1:21]
    Wt = test_data_w.values
    Yp = saved_model.predict(Xt)
    #Yp_unscaled = scaler.inverse_transform(Yp)
    layer_output_A = get_last_layer_output(tf.convert_to_tensor(Xt))[0]
    Yp2=lm_fit.predict(layer_output_A[:, (layer_output != 0).any(axis=0)])
    #Yp2=lm_fit.predict(layer_output_A[:, lii3])

    est=Yp2.T@Wt/Wt.sum()
    gamai=Yp2-est
    bias_array[j,0]=Yp.mean()-(true_mean)
    bias_array[j,1]=Yp2.mean()-(true_mean)
    est_array[j,0]=Yp.mean()
    est_array[j,1]=est
    ## Va
    na=Yp.size
    var_array[j,0]=1/na*(1-na/N)*np.var(gamai, ddof=1) + 1/na/(N-1)*sum(map(lambda i : i * i, gamai))
    ## Vb
    N_hat=Wt.sum()
    H=np.append(np.ones([len(df),1]),df,1)
    Hp = np.transpose(H) @ H
    #np.linalg.inv(Hp)
    if np.linalg.det(Hp)==0:
        sid=sid+1
        Hp_inv=np.linalg.pinv(Hp)
        sid_list.append(i)
    else:
        Hp_inv=np.linalg.inv(Hp)
    H_A=np.append(np.ones([len(layer_output_A),1]),layer_output_A[:, (layer_output != 0).any(axis=0)],1)
    tau=Hp_inv @ (np.transpose(H_A) @ Wt)
    eta=np.multiply((Y - Y_pred), (H @ tau).reshape(Y.size,1))
    eta_m=eta.sum()/N_hat
    var_array[j,1]=(np.square(eta-eta_m).sum()+eta_m*eta_m*(N-Y.size))/N_hat/(N_hat-1)
    ## V=Va+Vb
    var_array[j,2]=var_array[j,0]+var_array[j,1]
    # Export as CSV
    #full_pred_path = data_path + "pred_" + str(i) + ".csv"
    #df_Yp_unscaled = pd.DataFrame(data=Yp_unscaled)
    #df_Yp_unscaled.to_csv(full_pred_path,index=False)
    #if var_array[j,2]>5:
    #     break

bias_array.mean()
bias_array_nz = bias_array[(bias_array != 0).any(axis=1)]
print(np.mean(bias_array_nz,axis=0))
print(np.std(bias_array_nz,axis=0))
np.savetxt(fn_res, bias_array)
np.savetxt(fn_res2, est_array)
np.savetxt(fn_res3, var_array)
