sampA<-readRDS(file='bootci/survey_sub.rds')
sampB<-readRDS(file='bootci/blood_test_results.rds')

##subset based on agegroup
sampA=sampA[as.numeric(sampA$AGE_G)>=10,]
sampB=sampB[as.numeric(sampB$AGE_G)>=10,]

NB<-dim(sampB)[1]
nb<-5000
id_B<-sample(1:NB,nb)
sampB2<-sampB[id_B,]
sampA2<-sampA[is.na(sampA$HGB)=='FALSE' & is.na(sampA$TG)=='FALSE' & is.na(sampA$TCHOL)=='FALSE' & 
is.na(sampA$HDL)=='FALSE' & is.na(sampA$ANE)=='FALSE',]

#############functions for point estimation:
library(mgcv)
library(np)

sampA3<-sampA2[,c(4,1,11,2,3,5,6,7)]
sampA3$AGE_G<-as.numeric(sampA3$AGE_G)
sampA3$SEX<-as.numeric(sampA3$SEX)
sampA3$ANE<-as.numeric(sampA3$ANE)

sampB3<-sampB2[,c(4,1:3,5,6,7)]
sampB3$AGE_G<-as.numeric(sampB3$AGE_G)
sampB3$SEX<-as.numeric(sampB3$SEX)
sampB3$ANE<-as.numeric(sampB3$ANE)

sampA3$wt_hs


###Comparing the distributions of predictors:

##from A:
m1A<-sum(sampA3$wt_hs[sampA3$SEX==1])/sum(sampA3$wt_hs)
m2A<-sum(sampA3$AGE_G*sampA3$wt_hs)/sum(sampA3$wt_hs)
m3A<-sum(sampA3$HGB*sampA3$wt_hs)/sum(sampA3$wt_hs)
m4A<-sum(sampA3$TG*sampA3$wt_hs)/sum(sampA3$wt_hs)
m5A<-sum(sampA3$HDL*sampA3$wt_hs)/sum(sampA3$wt_hs)
m6A<-sum(sampA3$wt_hs[sampA3$ANE==1])/sum(sampA3$wt_hs)
mA<-c(m1A,m2A,m3A,m4A,m5A,m6A)

##from B:
nB<-dim(sampB3)[1]
wtB<-rep(1,nB)
sampB4<-cbind(sampB3,wtB)

m1B<-sum(sampB4$wtB[sampB4$SEX==1])/sum(sampB4$wtB)
m2B<-sum(sampB4$AGE_G*sampB4$wtB)/sum(sampB4$wtB)
m3B<-sum(sampB4$HGB*sampB4$wtB)/sum(sampB4$wtB)
m4B<-sum(sampB4$TG*sampB4$wtB)/sum(sampB4$wtB)
m5B<-sum(sampB4$HDL*sampB4$wtB)/sum(sampB4$wtB)
m6B<-sum(sampB4$wtB[sampB4$ANE==1])/sum(sampB4$wtB)

mB<-c(m1B,m2B,m3B,m4B,m5B,m6B)

MX<-cbind(mA,mB)
rownames(MX)<-c('SEX(Male)','AGE','HGB','TG','HDL', 'ANE')
colnames(MX)<-c('Mean A','Mean B')

write.csv(MX,file='MX.csv')

###1. The sample mean from sample A:
fM_A<-function(datA,datB){
etheta1<-sum(datA$TCHOL*datA$wt_hs)/sum(datA$wt_hs)
etheta2<-sum(datA$TCHOL[datA$SEX==1]*datA$wt_hs[datA$SEX==1])/sum(datA$wt_hs[datA$SEX==1])
etheta<-c(etheta1,etheta2)
return(etheta)
}

###2. The naive estimator (sample mean) from sample B:
fM_B<-function(datA,datB){
etheta1<-mean(datB$TCHOL)
etheta2<-mean(datB$TCHOL[datB$SEX==1])
etheta<-c(etheta1,etheta2)
return(etheta)
}

###4. The proposed nonparametric mass imputation estimator using GAM:
fGAM<-function(datA,datB){
XA<-datA[,2:7]
XA<-as.data.frame(XA)

###fit model by using sample B:
fit<-gam(TCHOL~ SEX + ANE + s(AGE_G)+s(HGB)+s(TG)+s(HDL),data=datB)
#fit<-gam(TCHOL~s(AGE_G)+s(HGB)+s(TG)+s(HDL),data=datB)

iyA<-predict(fit,XA)
etheta1<-sum(iyA*datA$wt_hs)/sum(datA$wt_hs)
etheta2<-sum(iyA[datA$SEX==1]*datA$wt_hs[datA$SEX==1])/sum(datA$wt_hs[datA$SEX==1])
etheta<-c(etheta1,etheta2)
return(etheta)
}

###5. The proposed nonparametric mass imputation estimator using LM:
fLM<-function(datA,datB){
XA<-datA[,2:7]
XA<-as.data.frame(XA)

###fit model by using sample B:
fit<-lm(TCHOL~SEX+ANE+AGE_G+HGB+TG+HDL,data=datB)
#fit<-lm(TCHOL~AGE_G+HGB+TG+HDL,data=datB)

iyA<-predict(fit,XA)
etheta1<-sum(iyA*datA$wt_hs)/sum(datA$wt_hs)
etheta2<-sum(iyA[datA$SEX==1]*datA$wt_hs[datA$SEX==1])/sum(datA$wt_hs[datA$SEX==1])
etheta<-c(etheta1,etheta2)
return(etheta)
}


library(dials)
library(recipes)
library(parsnip)
library(workflows)
library(rsample)
library(tidyverse)
library(tune)

fKNN=function(datA,datB){
 datB_tbl <- datB %>% as_tibble()
 datA_tbl <- datA %>% as_tibble()
 XA<-datA[,2:7]
 XA<-as.data.frame(XA)

    # * 0) recipe (COMMON) ----
    common_recipe <- recipe(
      TCHOL ~ ., data = datB_tbl
    )

      knn_spec <- nearest_neighbor(
                neighbors = tune(),
                weight_func = tune(),
                dist_power = tune()
        ) %>%
        set_mode("regression") %>%
        set_engine("kknn")

      # * 2) workflow ----
      knn_workflow <- workflow() %>%
        add_recipe(common_recipe) %>%
        add_model(knn_spec)

      # * 3) resamples ----
      #set.seed(123)
      dat_folds <- vfold_cv(datB_tbl,v = 10)

      # * 4) tune_grid() ----
      #set.seed(123)
      knn_cube <- grid_latin_hypercube(
        dist_power(),
        #neighbors(c(3,10),trans=NULL),
        neighbors(),
        weight_func(),
        size = 30
      )
      knn_grid <- tune_grid(
        knn_workflow,
        resamples = dat_folds,
        grid = knn_cube
      )

      #autoplot(knn_grid, metric = "rmse")

      # * 5) finalize_workflow() ----
      final_knn_workflow <- knn_workflow %>%
        finalize_workflow(select_best(knn_grid,metric = "rmse"))

      # * 6) last_fit() ----
      fit <- extract_spec_parsnip(final_knn_workflow) %>%
        fit(TCHOL ~ ., data = datB_tbl)

iyA<-predict(fit,XA)
iyA=iyA %>% pull(.pred)
etheta1<-sum(iyA*datA$wt_hs)/sum(datA$wt_hs)
etheta2<-sum(iyA[datA$SEX==1]*datA$wt_hs[datA$SEX==1])/sum(datA$wt_hs[datA$SEX==1])
etheta<-c(etheta1,etheta2)
return(etheta)

}

fXGBoost=function(datA,datB){
 datB_tbl <- datB %>% as_tibble()
 XA<-datA[,2:7]
 XA<-as.data.frame(XA)

    # * 0) recipe (COMMON) ----
    common_recipe <- recipe(
      TCHOL ~ ., data = datB_tbl
    )

      # * 1) model spec xgboost ----
      boost_spec <- boost_tree(
        tree_depth = tune(),
        trees = tune(),              # add tune() here as well - 20, 50, 100, 200, 300, etc.
        learn_rate = tune(),
        mtry = tune(),
        min_n = tune(),
        #loss_reduction = tune(),
        sample_size = 0.8
      ) %>%
        set_mode("regression") %>%
        set_engine("xgboost")

      # * 2) workflow ----
      boost_workflow <- workflow() %>%
        add_recipe(common_recipe) %>%
        add_model(boost_spec)

      # * 3) resamples ----
      #set.seed(123)
      dat_folds <- vfold_cv(datB_tbl,v = 10)

      # * 4) tune_grid() ----
      boost_cube <- grid_latin_hypercube(
        tree_depth(),
        trees(),
        learn_rate(),
        finalize(mtry(), datB_tbl),
        min_n(),
        #loss_reduction(),
        #sample_size = sample_prop(),
        size = 30
      )

      #set.seed(123)
      boost_grid <- tune_grid(
        boost_workflow,
        resamples = dat_folds,
        grid = boost_cube,
        control = control_grid(save_pred = T)
      )

        #autoplot(boost_grid, metric = "rmse")

      # * 5) finalize_workflow() ----
      final_boost_workflow <- boost_workflow %>%
        finalize_workflow(select_best(boost_grid,metric = "rmse"))


# * 6) last_fit() ----
      fit <- pull_workflow_spec(final_boost_workflow) %>%
        fit(TCHOL ~ ., data = datB_tbl)

iyA<-predict(fit,XA)
iyA=iyA %>% pull(.pred)
etheta1<-sum(iyA*datA$wt_hs)/sum(datA$wt_hs)
etheta2<-sum(iyA[datA$SEX==1]*datA$wt_hs[datA$SEX==1])/sum(datA$wt_hs[datA$SEX==1])
etheta<-c(etheta1,etheta2)
return(etheta)

}

fRT=function(datA,datB){
 datB_tbl <- datB %>% as_tibble()
 XA<-datA[,2:7]
 XA<-as.data.frame(XA)

    # * 0) recipe (COMMON) ----
    common_recipe <- recipe(
      TCHOL ~ ., data = datB_tbl
    )

      # * 1) model spec rpart ----
      rpart_spec <- decision_tree(
        tree_depth = tune(),
        min_n = tune(),
        cost_complexity = tune()
      ) %>%
        set_mode("regression") %>%
        set_engine("rpart")

      # * 2) workflow ----
      rpart_workflow <- workflow() %>%
        add_recipe(common_recipe) %>%
        add_model(rpart_spec)

      # * 3) resamples ----
      #set.seed(123)
      dat_folds <- vfold_cv(datB_tbl,v = 10)

      # * 4) tune_grid() ----
      #set.seed(123)
      rpart_cube <- grid_latin_hypercube(
        cost_complexity(),
        min_n(),
        tree_depth(),
        size = 30
      )
      rpart_grid <- tune_grid(
        rpart_workflow,
        resamples = dat_folds,
        grid = rpart_cube
      )
      #autoplot(rpart_grid, metric = "rmse")

      # * 5) finalize_workflow() ----
      final_rpart_workflow <- rpart_workflow %>%
        finalize_workflow(select_best(rpart_grid,metric = "rmse"))

      # * 6) last_fit() ----
      fit <- pull_workflow_spec(final_rpart_workflow) %>%
        fit(TCHOL ~ ., data = datB_tbl)

iyA<-predict(fit,XA)
iyA=iyA %>% pull(.pred)
etheta1<-sum(iyA*datA$wt_hs)/sum(datA$wt_hs)
etheta2<-sum(iyA[datA$SEX==1]*datA$wt_hs[datA$SEX==1])/sum(datA$wt_hs[datA$SEX==1])
etheta<-c(etheta1,etheta2)
return(etheta)

}


RES_P_A<-fM_A(sampA3,sampB3)
RES_P_B<-fM_B(sampA3,sampB3)
RES_P_PMIE<-fP_A(sampA3,sampB3)

RES_P_NPMIEG<-fGAM(sampA3,sampB3)
RES_P_LM<-fLM(sampA3,sampB3)
RES_P_XGB<-fXGBoost(sampA3,sampB3)
RES_P_RT<-fRT(sampA3,sampB3)
RES_P_KNN<-fKNN(sampA3,sampB3)

RES_P<-rbind(RES_P_B-RES_P_A,RES_P_PMIE-RES_P_A,RES_P_PWE-RES_P_A,RES_P_NPMIEK-RES_P_A,
RES_P_NPMIEG-RES_P_A)

rbind(RES_P_NPMIEG,RES_P_LM,RES_P_XGB,RES_P_RT,RES_P_KNN)

colnames(RES_P)<-c('Mean','Domain mean')
rownames(RES_P)<-c('Mean B','PMIE','PWE','NPMIEK','NPMIEG')

#########################################Variance estimation:

###Bootstrap variance estimation:
library(survey)

B_B<-500
Theta0<-fM_A(sampA3,sampB3)

RES_VAR<-fboot(sampA3,sampB3,Theta0,500)

fboot<-function(datA,datB,theta0,B_B){
nA<-dim(datA)[1]
nB<-dim(datB)[1]
d=datA$wt_hs #design weights
eN=sum(d) # pop size
prob=d/sum(d)

#etheta0_P_A<-fP_A(datA,datB)
etheta0_GAM<-fGAM(datA,datB)
etheta0_LM<-fLM(datA,datB)
etheta0<-c(etheta0_GAM,etheta0_LM)

###bootstrap procedure:
S_etheta<-NULL
iter<-0
repeat{
iter<-iter+1
##bootstrap sample from B:
s_id_B<-sample(1:nB,nB,replace=T)
datBb<-datB[s_id_B,]

##bootstrap sample from A:

#1 select boostrap pseudo population
#cat("b=",b,"\n")
U=sample(1:nA, eN, prob=prob, replace=T)

#2 choose boostrap sample
Pi=1/d[U] #inclusion probabilites
Ub.in=rbinom(eN,1,Pi) #inclusion indicators
Ub=U[Ub.in==1] #selecet bootstrap sample
datAb<-datA[Ub,]

#s_etheta_P_A<-fP_A(datAb,datBb)
s_etheta_GAM<-fGAM(datAb,datBb)
s_etheta_LM<-fLM(datAb,datBb)
s_etheta<-c(s_etheta_GAM,s_etheta_LM)

S_etheta<-cbind(S_etheta,s_etheta)

if(iter==B_B){break}
}

eV<-apply((S_etheta-etheta0)^{2},1,mean,na.rm=T)

qz<-qnorm(0.975)

LB<-etheta0-qz*sqrt(eV)
UB<-etheta0+qz*sqrt(eV)
AL<-UB-LB
CI<-as.numeric(LB<theta0 & theta0<UB)

return(c(eV,AL,CI))
}

write.table(sampA3, "sampA3.csv", row.names=F, sep=",")
write.table(sampB3, "sampB3.csv", row.names=F, sep=",")

