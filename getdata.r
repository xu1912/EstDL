library(tidyverse)
pacman::p_load(
  tidyverse,
  tidymodels,
  tidyquant,
  mgcv,
  np,
  survey,
  sampling
  # torch,  # Uncomment to install! Once installed, the mlverse tabnet demo doesn't call the torch library (it's like those other libraries used in tidymodels engines)
  # tabnet
)

fM_A <- function(dat){
  y       <- dat[,1]
  x1      <- dat[,2]
  sIA     <- dat[,6]
  syA     <- y[sIA == 1]
  sx1A    <- x1[sIA == 1]

  etheta1 <- mean(syA)
  etheta2 <- median(syA)
  etheta  <- c(etheta1,etheta2)

  return(etheta)
}

fM_B <- function(dat){

  y       <- dat[,1]
  x1      <- dat[,2]
  sIB     <- dat[,7]
  syB     <- y[sIB==1]
  sx1B    <- x1[sIB==1]

  etheta1 <- mean(syB)
  etheta2 <- median(syB)
  etheta  <- c(etheta1,etheta2)

  return(etheta)
}

# <><> gedata (generation of simulated data) ----
# gedata
## fsA
## fsrs
# OLD: B = 1000; N = 10000; nA = 500; nB = 1000; id_m = 1,2,3, or 4
# NEW: C = the controlling parameter for the sample size of sample B
## C = -.4.4 for Models 1-3 (n ~ 500); C = -20 for Model 4 (n ~ 500)
gedata_modified <- function(B,N,nA,C,id_m, err_term="n"){

  # Super population model ----
  # set.seed(123) # IMPORTANT: This won't make all datasets equal. If B = 1000 then you'll be able to make 1000 different datasets.

  x1 <- runif(B*N,-1,1)
  x2 <- runif(B*N,-1,1)
  x3 <- runif(B*N,-1,1)
  x4 <- runif(B*N,-1,1)

  if(err_term=="e"){
  #  epsilon <- rexp(B*N) - 1
     epsilon <- rexp(B*N, rate=3.3) - 1/3.3
  }else{
    epsilon <- rnorm(B*N, sd=0.3)
  }

  # For index == 1 ----
  if (id_m == 1){
    y <- 1 + -0.9149491*x1 + 0.3575822*x2 + 0.4353772*x3 + -0.4569431*x4 + epsilon
  }

  # For index == 2 ----
  if (id_m == 2){
   # y <- 1 + 1.481344*x1^2 + -1.860457*x2^2 + 1.978122*x3^2 + -1.713827*x4^2 + epsilon
    x1 <- runif(B*N,1,3)
    x2 <- runif(B*N,1,3)
    x3 <- runif(B*N,1,3)
    x4 <- runif(B*N,1,3)

    y <- 3 + 0.2*x1*x2*x3^3 + 0.3*x3*x4  + rnorm(B*N, sd=1)

  }

  if (id_m == 3){
    #y <- 1 + -0.1641609*abs(x1) + -0.4648368*x2*x3 + -0.3519404*x3*x4^2 + 0.8642934*x1*x2*x4 + epsilon
    y <- 1 + -0.1641609*abs(x1) + -0.4648368*x2*x3 + -0.3519404*x3*x4^2 + 0.8642934*x1*x2*x4 + rnorm(B*N, sd=0.1)
    #y <- 1 + -0.1641609*x1^2 + -0.4648368*x1*x2*x3 + -0.3519404*x1*x2 + 0.8642934*x1*x2*x3*(x4^2) + rnorm(B*N, sd=0.1)
  }

  # For index == 4 ----
  if (id_m == 4){
    range_min <- -2
    range_max <- 2

    if(err_term=="e"){
      epsilon <- rexp(B*N, rate=3.3) - 1/3.3
    }else{
      epsilon <- rnorm(B*N, sd=0.3)
    }
    #beta_0_thru_3 <- runif(4,range_min,range_max)
    beta_0_thru_3 = c(-1.5719112, -1.2716745, 1.1025010, -0.3454468)

    #alpha_1.1 <- runif(21,range_min,range_max)
    #alpha_2.1 <- runif(21,range_min,range_max)
    #alpha_3.1 <- runif(21,range_min,range_max)

    alpha_1.1 <- c(1.62518936023116,-0.988645439036191,0.781700223684311,1.35625289939344,0.0915707796812057,-1.49375763628632,1.34448341745883,-1.33983169030398,-1.48179383762181,0.422844471409917,1.2425712896511,1.21518217399716,0.220892492681742,-1.88863917067647,-0.472173160873353,1.00099825207144,-0.0778127191588283,-0.93792312219739,-1.11116895917803,1.30368058662862,-1.10556009877473)
    alpha_2.1 <- c(1.30507991090417,-1.86457681003958,1.96857705339789,-0.218055023811758,-1.29031549580395,1.88011801894754,-0.805277395993471,-1.19949021283537,1.20774274505675,0.89913238491863,-1.79890841431916,-1.04121180810034,-1.38355937320739,1.20570533908904,-1.24887393042445,-0.116144786588848,-0.345157947391272,1.63900570105761,-0.151626331731677,-0.6426852485165,-1.33341608196497)
    alpha_3.1 <- c(-0.681133939884603,-0.144093198701739,-1.33295672945678,-0.634271279908717,0.760696358047426,1.65304967388511,-1.25418273266405,0.400921450927854,-0.966644627973437,0.167373306117952,-0.944373677484691,0.471613362431526,-1.95782957039773,1.9849669020623,0.414783649146557,-1.94344886764884,-0.252952396869659,1.6570187387988,0.770739193074405,0.950334971770644,0.599851266480982)

    x1 <- runif(B*N,-1,1)
    x2 <- runif(B*N,-1,1)
    x3 <- runif(B*N,-1,1)
    x4 <- runif(B*N,-1,1)
    x5 <- runif(B*N,-1,1)

    x6 <- runif(B*N,-1,1)
    x7 <- runif(B*N,-1,1)
    x8 <- runif(B*N,-1,1)
    x9 <- runif(B*N,-1,1)
    x10 <- runif(B*N,-1,1)

    x11 <- runif(B*N,-1,1)
    x12 <- runif(B*N,-1,1)
    x13 <- runif(B*N,-1,1)
    x14 <- runif(B*N,-1,1)
    x15 <- runif(B*N,-1,1)

    x16 <- runif(B*N,-1,1)
    x17 <- runif(B*N,-1,1)
    x18 <- runif(B*N,-1,1)
    x19 <- runif(B*N,-1,1)
    x20 <- runif(B*N,-1,1)

    # First layer
    a_1.1 <- log( 1 + exp( alpha_1.1[1] + alpha_1.1[2]*x1 + alpha_1.1[3]*x2 + alpha_1.1[4]*x3 + alpha_1.1[5]*x4 + alpha_1.1[6]*x5 + alpha_1.1[7]*x6 + alpha_1.1[8]*x7 + alpha_1.1[9]*x8 + alpha_1.1[10]*x9 + alpha_1.1[11]*x10 + alpha_1.1[12]*x11 + alpha_1.1[13]*x12 + alpha_1.1[14]*x13 + alpha_1.1[15]*x14 + alpha_1.1[16]*x15 + alpha_1.1[17]*x16 + alpha_1.1[18]*x17 + alpha_1.1[19]*x18 + alpha_1.1[20]*x19 + alpha_1.1[21]*x20 ) )
    a_2.1 <- log( 1 + exp( alpha_2.1[1] + alpha_2.1[2]*x1 + alpha_2.1[3]*x2 + alpha_2.1[4]*x3 + alpha_2.1[5]*x4 + alpha_2.1[6]*x5 + alpha_2.1[7]*x6 + alpha_2.1[8]*x7 + alpha_2.1[9]*x8 + alpha_2.1[10]*x9 + alpha_2.1[11]*x10 + alpha_2.1[12]*x11 + alpha_2.1[13]*x12 + alpha_2.1[14]*x13 + alpha_2.1[15]*x14 + alpha_2.1[16]*x15 + alpha_2.1[17]*x16 + alpha_1.1[18]*x17 + alpha_2.1[19]*x18 + alpha_2.1[20]*x19 + alpha_2.1[21]*x20 ) )
    a_3.1 <- log( 1 + exp( alpha_3.1[1] + alpha_3.1[2]*x1 + alpha_3.1[3]*x2 + alpha_3.1[4]*x3 + alpha_3.1[5]*x4 + alpha_3.1[6]*x5 + alpha_3.1[7]*x6 + alpha_3.1[8]*x7 + alpha_3.1[9]*x8 + alpha_3.1[10]*x9 + alpha_3.1[11]*x10 + alpha_3.1[12]*x11 + alpha_3.1[13]*x12 + alpha_3.1[14]*x13 + alpha_3.1[15]*x14 + alpha_3.1[16]*x15 + alpha_3.1[17]*x16 + alpha_3.1[18]*x17 + alpha_3.1[19]*x18 + alpha_3.1[20]*x19 + alpha_3.1[21]*x20 ) )

    alpha_1.2 <- c(-0.234935183078051,0.813478100113571,1.8260705191642,1.38791483268142)
    alpha_2.2 <- c(1.58547177258879,-0.797669362276793,-1.02610027045012,-1.94539337418973)
    alpha_3.2 <- c(0.911239068955183,-1.98296133801341,1.78652271814644,0.932876035571098)

    # Second layer
    a_1.2 <- log( 1 + exp( alpha_1.2[1] + alpha_1.2[2]*a_1.1 + alpha_1.2[3]*a_2.1 + alpha_1.2[4]*a_3.1 ) )
    a_2.2 <- log( 1 + exp( alpha_2.2[1] + alpha_2.2[2]*a_1.1 + alpha_2.2[3]*a_2.1 + alpha_2.2[4]*a_3.1 ) )
    a_3.2 <- log( 1 + exp( alpha_3.2[1] + alpha_3.2[2]*a_1.1 + alpha_3.2[3]*a_2.1 + alpha_3.2[4]*a_3.1 ) )

    # y = third layer
    y <- beta_0_thru_3[1] + beta_0_thru_3[2]*a_1.2 + beta_0_thru_3[3]*a_2.2 + beta_0_thru_3[4]*a_3.2 + epsilon
  }
  M  <- rep(1:N,B)
  M2 <- matrix(M,B,N,byrow=T)

  # Function for SRS ----
  fsA <- function(a_large_matrix_row){            # That is, a large matrix row of integers: 1:10000
    sI0      <- rep(0,length(a_large_matrix_row)) # A vector of zeros
    id0      <- sample(a_large_matrix_row,nA)     # That way this is a random index, get it?
    sI0[id0] <- 1                                 # So the vector of zeros can be indexed randomly!
    return(sI0)
  }

  # This is where M2 is passed to fsA() one row at a time
  MsIA0 <- apply(M2,1,fsA)
  sIA   <- as.numeric(MsIA0)

  if (id_m %in% c(1,2,3)){
    ## M1-M3 sample B ----
    ### Experiment with c = -4, something else, to get a good proportion of the data--about 500.
    rho=c(0.8174850, -1.8716409, -1.1715428, -0.3204672)
    C=-2.8
    p   <- exp(C + rho[1]*x1+rho[2]*x2+rho[3]*x3+rho[4]*x4)/(1+exp(C + rho[1]*x1+rho[2]*x2+rho[3]*x3+rho[4]*x4))
    sIB <- rbinom(B*N,1,p) # 'p' is only used here. sIB is obviously used below.
  }

  if (id_m %in% c(2)){
    ## M1-M3 sample B ----
    ### Experiment with c = -4, something else, to get a good proportion of the data--about 500.
    rho=c(0.8174850, -1.8716409, -1.1715428, -0.3204672)
    C=2.24
    p   <- exp(C + rho[1]*x1+rho[2]*x2+rho[3]*x3+rho[4]*x4)/(1+exp(C + rho[1]*x1+rho[2]*x2+rho[3]*x3+rho[4]*x4))
    sIB <- rbinom(B*N,1,p) # 'p' is only used here. sIB is obviously used below.
  }


  if (id_m == 4){
    #rho <- runif(3,min = 1,max = 3)
    rho=c(1.0206762,-0.7120927, 1.6224493)
    ## M4 sample B ----
    ### Experiment with c = 1, etc. until you get sample B's n = ~500.
    C=-6.8
    p   <- exp(C + rho[1]*a_1.1+rho[2]*a_2.1+rho[3]*a_3.1) /
      ( 1 + exp(C + rho[1]*a_1.1+rho[2]*a_2.1+rho[3]*a_3.1) )
    #p   <- exp(C + x1+x2+x3+x4)/(1+exp(C + x1+x2+x3+x4))
    sIB <- rbinom(B*N,1,p)
  }

  w     <- rep((N/nA),(B*N))

  if (id_m %in% c(1,2,3)) {
    # Create the matrix of 7 columns of simulated data! Recall that these are 10 million elements each.
    My   <- matrix(y,B,N,byrow=T)
    Mx1  <- matrix(x1,B,N,byrow=T)
    Mx2  <- matrix(x2,B,N,byrow=T)

    # ADD Mx3 ----
    Mx3  <- matrix(x3,B,N,byrow=T)

    # ... ADD Mx4 ----
    Mx4  <- matrix(x4,B,N,byrow=T)

    MsIA <- matrix(sIA,B,N,byrow=T)
    MsIB <- matrix(sIB,B,N,byrow=T)
    Mw   <- matrix(w,B,N,byrow=T)

    M   <- cbind(My,  # will become dat[,1]
                 Mx1, # will become dat[,2]
                 Mx2, # will become dat[,3]

                 # ADD Mx3 ----
                 Mx3, # will become dat[,4]

                 # ... ADD Mx4  ----
                 Mx4, # will become dat[,5]

                 MsIA,# will become dat[,6]
                 MsIB,# will become dat[,7]
                 Mw)  # will become dat[,8]
    M   <- t(M)
    aM  <- as.numeric(M)

    # ... ADD 8 to Res so Mx4 fits ----
    # If you take 10 million elements and make 1000 matrices out of it, there will be 10,000 rows in each matrix.
    Res <- array(aM,c(N,8,B))
  }
    # Create the matrix of 7 columns of simulated data! Recall that these are 10 million elements each.
    My   <- matrix(y,B,N,byrow=T)
    Mx1  <- matrix(x1,B,N,byrow=T)
    Mx2  <- matrix(x2,B,N,byrow=T)

    # ADD Mx3 ----
    Mx3  <- matrix(x3,B,N,byrow=T)

    # ... ADD 4.thru.21 ----
    Mx4 <- matrix(x4,B,N,byrow=T)
    Mx5 <- matrix(x5,B,N,byrow=T)
    Mx6 <- matrix(x6,B,N,byrow=T)
    Mx7 <- matrix(x7,B,N,byrow=T)
    Mx8 <- matrix(x8,B,N,byrow=T)
    Mx9 <- matrix(x9,B,N,byrow=T)
    Mx10 <- matrix(x10,B,N,byrow=T)
    Mx11 <- matrix(x11,B,N,byrow=T)
    Mx12 <- matrix(x12,B,N,byrow=T)
    Mx13 <- matrix(x13,B,N,byrow=T)
    Mx14 <- matrix(x14,B,N,byrow=T)
    Mx15 <- matrix(x15,B,N,byrow=T)
    Mx16 <- matrix(x16,B,N,byrow=T)
    Mx17 <- matrix(x17,B,N,byrow=T)
    Mx18 <- matrix(x18,B,N,byrow=T)
    Mx19 <- matrix(x19,B,N,byrow=T)
    Mx20 <- matrix(x20,B,N,byrow=T)

    MsIA <- matrix(sIA,B,N,byrow=T)
    MsIB <- matrix(sIB,B,N,byrow=T)
    Mw   <- matrix(w,B,N,byrow=T)
    M   <- cbind(My,   # will become dat[,1]
                 Mx1,  # will become dat[,2]
                 Mx2,  # will become dat[,3]

                 # ADD Mx3 ----
                 Mx3,  # will become dat[,4]

                 # ... ADD 4.thru.21 ----
                 Mx4,  # will become dat[,5]
                 Mx5,  # will become dat[,6]
                 Mx6,  # will become dat[,7]
                 Mx7,  # will become dat[,8]
                 Mx8,  # will become dat[,9]
                 Mx9,  # will become dat[,10]
                 Mx10, # will become dat[,11]
                 Mx11, # will become dat[,12]
                 Mx12, # will become dat[,13]
                 Mx13, # will become dat[,14]
                 Mx14, # will become dat[,15]
                 Mx15, # will become dat[,16]
                 Mx16, # will become dat[,17]
                 Mx17, # will become dat[,18]
                 Mx18, # will become dat[,19]
                 Mx19, # will become dat[,20]
                 Mx20, # will become dat[,21]

                 MsIA, # will become dat[,22]
                 MsIB, # will become dat[,23]
                 Mw)   # will become dat[,24]
    M   <- t(M)
    aM  <- as.numeric(M)

    # ... ADD 24 to Res so Mx4.thru.21 fits ----
    # If you take 10 million elements and make 1000 matrices out of it, there will be 10,000 rows in each matrix.
    Res <- array(aM,c(N,24,B))
  }

  # Population mean of Y
  theta0_1 <- mean(y)
  #theta0_2 <- median(y)

  res <- list(Res,theta0_1)

  return(res)
}

FRES <- function(indat,modeling_method = "GAM",id_m){

  dat    <- indat[[1]]
  THETA0 <- c(indat[[2]])  ##,indat[[3]])
  res_GAM    <- apply(dat,3,f_ML,
                      modeling_method = modeling_method,
                      id_m = id_m)
  bias_GAM   <- res_GAM - THETA0
  m_bias_GAM <- apply(bias_GAM,1,mean)
  rb_GAM     <- m_bias_GAM / THETA0             # rb_GAM
  var_GAM    <- apply(res_GAM,1,var)
  se_GAM     <- sqrt(var_GAM)
  rse_GAM    <- se_GAM / THETA0                 # rse_GAM
  mse_GAM    <- m_bias_GAM^2 + var_GAM
  rrmse_GAM  <- sqrt(mse_GAM) / THETA0          # rrmse_GAM
  Res_GAM    <- cbind(rb_GAM,rse_GAM,rrmse_GAM) # Res_GAM

  RES <- rbind(
    #Res_MA,
    #Res_MB,
    Res_GAM
  )
  RES <- round(RES,4)

  rownames(RES) <- c(
    #c('Mean_A','Median_A'),
    #c('Mean_B','Median_B'),
    # c('Mean_P','Domain Mean_P'),
    # c('Mean_PW','Domain Mean_PW'),
    # c('Mean_KS','Domain Mean_KS'),
    c('Mean_ML','Median_ML')
  )

  colnames(RES) <- c('RB','RSE','RRMSE')

  return(RES)
}
