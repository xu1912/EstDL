from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications, optimizers
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import importlib.metadata
import dask.array as da
from dask_glm.estimators import LinearRegression
from dask_glm.regularizers import L1
import optuna

# Define the model as a function
def create_model(hidden_units=32, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=20, activation='relu'))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

# Define the model as a function
def create_model2(learning_rate=0.001):
    model = Sequential()
    model.add(Dense(1, input_dim=RX.shape[1], activation='linear',use_bias=False))
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


# Define the objective function for Optuna
def objective(trial):
    hidden_units = trial.suggest_int("hidden_units", 50, 100)
    learning_rate = trial.suggest_float("learning_rate",  1e-4, 1e-2, log=True)

    # Create the Keras model
    model = create_model(hidden_units=hidden_units, learning_rate=learning_rate)

    # Define cross-validation and EarlyStopping
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Perform cross-validation
    val_scores = []

    for train_index, val_index in kfold.split(RX):
        X_train, X_val = RX[train_index], RX[val_index]
        y_train, y_val = Ry[train_index], Ry[val_index]
        
        # Fit the model with EarlyStopping
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,
                            batch_size=32, callbacks=[early_stopping],verbose=0)
        
        # Evaluate on the validation set
        val_score =model.evaluate(X_val, y_val)[0]
        val_scores.append(val_score)
    
    # Return the mean validation score
    return np.mean(val_scores)

# Define the objective function for Optuna
def objective2(trial):
    learning_rate = trial.suggest_float("learning_rate",  1e-5, 1e-2, log=True)

    # Create the Keras model
    model = create_model2( learning_rate=learning_rate)

    # Define cross-validation and EarlyStopping
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Perform cross-validation
    val_scores = []

    for train_index, val_index in kfold.split(RX):
        X_train, X_val = RX[train_index], RX[val_index]
        y_train, y_val = Ry[train_index], Ry[val_index]
        
        # Fit the model with EarlyStopping
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,
                            batch_size=32, callbacks=[early_stopping],verbose=0)
        
        # Evaluate on the validation set
        val_score =model.evaluate(X_val, y_val)[0]
        val_scores.append(val_score)
    
    # Return the mean validation score
    return np.mean(val_scores)

# Make sure the path is specific to test 3, 4, 5; selu 3, 4, 5 !!!!!
data_path = "C:/Users/cxu2/Documents/EstDL/dat4_n_rev/"
ns=10
N=10000
bias_array = np.zeros( (ns, 2) )
est_array = np.zeros( (ns, 2) )
var_array = np.zeros( (ns, 3) )
true_mean=4.401465            #4n: -11.69074; 4e: -1.483135; 3n: 0.9452195; 3e: 2.000757; 1n: 0.9997838; 1e: 0.9994008; 
true_mean=0.5516707
#1n-2k: 0.9999008;2n-2k: 12.19999; 4n-2k: -11.69195

#true_median=0.9990163          #4n: -0.6511648; 4e: -1.395075; 3n: 1.619938; 3e: 1.439664; 1n: 0.9990163; 1e: 0.9448088;
fn_res = data_path + "cres_3L_150n_lr01_bias.txt"
fn_res2 = data_path + "cres_3L_150n_lr01_est.txt"
fn_res3 = data_path + "cres_3L_150n_lr01_var.txt"

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
    #scaler = StandardScaler()
    new_columns=dat2.columns.values
    new_columns[0]="syB"
    dat2.columns=new_columns
    X = dat2.drop('syB', axis=1).values
    Y = dat2[['syB']].values
    #scaler.fit(Y)
    #Y_scaled = scaler.transform(Y)
    # Y
    # scaler.inverse_transform(Y_scaled)
    # Scale both the training inputs and outputs
    # scaled_training = scaler.fit_transform(training_data_df)
    # scaled_testing = scaler.transform(test_data_df)
    # Define the model
    # ADD / SUBTRACT number of layers (try 2, 3, 4 output included)
    model = Sequential()
    model.add(Dense(150, input_dim=20, activation='relu'))
    #model.add(Dense(150, activation='relu'))
    model.add(Dense(150, activation='relu'))
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
    
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    num_variables = sum([tf.size(var).numpy() for var in saved_model.trainable_variables]) 
    gradient_matrix = np.zeros((X.shape[0], num_variables))

    for ii in range(X.shape[0]): 
        with tf.GradientTape() as single_tape: 
            single_prediction = saved_model(X[ii:ii+1]) # Get a single prediction 
            grads = single_tape.gradient(single_prediction, saved_model.trainable_variables) 
            # Flatten and concatenate gradients to form a single row in the matrix 
            gradient_row = np.concatenate([tf.reshape(grad, [-1]).numpy() for grad in grads]) 
            gradient_matrix[ii, :] = gradient_row
    
    res_r=gradient_matrix.T@gradient_matrix
   
    full_A_path = data_path + "dtA_" + str(i) + ".csv"
    full_Aw_path = data_path + "dtAw_" + str(i) + ".csv"
    test_data = pd.read_csv(full_A_path)
    test_data_w = pd.read_csv(full_Aw_path)
    Xt = test_data.values[:,1:21]
    Wt = test_data_w.values
    Yp = saved_model.predict(Xt)
    
    gradient_matrix_a = np.zeros((Xt.shape[0], num_variables))

    for ii in range(Xt.shape[0]): 
        with tf.GradientTape() as single_tape: 
            single_prediction = saved_model(Xt[ii:ii+1]) # Get a single prediction 
            grads = single_tape.gradient(single_prediction, saved_model.trainable_variables) 
            # Flatten and concatenate gradients to form a single row in the matrix 
            gradient_row = np.concatenate([tf.reshape(grad, [-1]).numpy() for grad in grads]) 
            gradient_matrix_a[ii, :] = gradient_row

    res_a=gradient_matrix_a.T@Wt

    A=res_r/N
    B=res_a/N

    RX = A # 100 samples, 10 features 
    Ry = B # 100 target values # Split the data into training and testing sets 
    Ry = Ry.ravel()
    
    large_array_X = da.from_array(RX, chunks=(25951, 25951))
    large_array_y = da.from_array(Ry, chunks=(25951,))
    
    lr = LinearRegression(regularizer=L1())
    lr.fit(large_array_X, large_array_y)
                                  
    lasso_cv = LassoCV(cv=10, random_state=42)
    lasso_cv.fit(large_array, Ry)
    optimal_lambda = lasso_cv.alpha_
    coefficients = lasso_cv.coef_
    
    # Define the study and start optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective2, n_trials=10)

    model_tau = Sequential()
    model_tau.add(Dense(units=1, input_dim=RX.shape[1], activation='linear', use_bias=False))  # Setting use_bias=False removes the intercept

    # Compile the model
    model_tau.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mean_squared_error'])
    es = EarlyStopping(monitor='mean_squared_error', mode='min', verbose=1, patience=10)
    fn2_best_model='best_model2' +'.h5'
    mc = ModelCheckpoint(fn2_best_model, monitor='mean_squared_error', mode='min', verbose=1, save_best_only=True)
    # Train the model
    model_tau.fit(RX, Ry, epochs=40, callbacks = [es, mc], verbose=0)
    saved_model2 = load_model(fn2_best_model)

    # Print the model parameters (weights only, as bias is set to False)
    weights_tau = saved_model2.layers[0].get_weights()[0]

    gmm=gradient_matrix@weights_tau
    predictions = saved_model(X)
    phi=gmm*(Y-predictions)
    N_hat=Wt.sum()
    phi_mean=sum(phi)/N_hat
    ## Vb
    var_array[j,1]=(sum(phi*phi) - sum(2*phi*phi_mean) + N_hat*phi_mean**2)/N_hat/(N_hat-1)

    est=Yp.T@Wt/Wt.sum()
    gamai=Yp-est
    bias_array[j,0]=Yp.mean()-(true_mean)
    est_array[j,0]=Yp.mean()
    est_array[j,1]=est
    ## Va
    na=Yp.size
    var_array[j,0]=1/na*(1-na/N)*np.var(gamai, ddof=1) + 1/na/(N-1)*sum(map(lambda i : i * i, gamai))
    
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
