f_ML <- function(dat,modeling_method,id_m){

  # . ----
  # FOR EQUATIONS 1-3 ----
  if(id_m %in% c(1,2,3)){

    # M   <- cbind(My,  # will become dat[,1]
    #              Mx1, # will become dat[,2]
    #              Mx2, # will become dat[,3]
    #
    #              # ADD Mx3 ----
    #              Mx3, # will become dat[,4]
    #
    #              # ... ADD Mx4  ----
    #              Mx4, # will become dat[,5]
    #
    #              MsIA,# will become dat[,6]
    #              MsIB,# will become dat[,7]
    #              Mw)  # will become dat[,8]
    y      <- dat[,1]
    x1     <- dat[,2]
    x2     <- dat[,3]

    # Weight
    Mw     <- dat[,8]

    # ADD x3
    x3     <- dat[,4]

    # ... ADD Mx4  ----
    x4     <- dat[,5]

    # Sample Indicators for A and B
    sIA    <- dat[,6]
    sIB    <- dat[,7]

    #
    swA   <- Mw[sIA == 1]
    swB   <- Mw[sIB == 1]

    #
    sx1A   <- x1[sIA == 1]
    sx1B   <- x1[sIB == 1]

    #
    sx2A   <- x2[sIA == 1]
    sx2B   <- x2[sIB == 1]
    sx3A   <- x3[sIA == 1]
    sx3B   <- x3[sIB == 1]

    # ... ADD sx4A and B ----
    sx4A   <- x4[sIA == 1]
    sx4B   <- x4[sIB == 1]

    #
    syB    <- y[sIB == 1]

    # ... ADD sx4B to datB ----
    datB   <- cbind(syB,sx1B,sx2B,sx3B,sx4B)
    datB2  <- as.data.frame(datB)

    # ... ADD sx4A to datXA ----
    datXA  <- cbind(sx1A,sx2A,sx3A,sx4A)
    datXA2 <- as.data.frame(datXA)

    # The colnames for A are labelled with B's because we're gonna predict using A as new data
    # (has to have same column names)
    colnames(datXA2) <- c('sx1B','sx2B','sx3B','sx4B')

    # ADD tibbles for datB2 & datXA2 ----
    datB2_tbl <- datB2 %>% as_tibble()
    datXA2_tbl <- datXA2 %>% as_tibble()

    # * 0) recipe (COMMON) ----
    common_recipe <- recipe(
      syB ~ ., data = datB2_tbl
    )

    # Fit model to sample B ----
    if (modeling_method        == "GAM") {
      fit <- gam(syB ~ s(sx1B) + s(sx2B) + s(sx3B) + s(sx4B), # ... GAM's x4 ----
                 data = datB2)
    }

    if (modeling_method == "LM") {
      fit <- linear_reg() %>%
        set_engine("lm") %>%
        fit(syB ~ sx1B + sx2B + sx3B + sx4B, # ... LM's x4 ----
            data = datB2)
    }
    if (modeling_method == "KNN") {
      # * 1) model spec rpart ----
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
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl,v = 10)

      # * 4) tune_grid() ----
      set.seed(123)
      knn_cube <- grid_latin_hypercube(
        dist_power(),
        neighbors(c(3,10),trans=NULL),
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
        fit(syB ~ ., data = datB2_tbl)

    }
    if (modeling_method == "XGBOOST") {

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
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl,v = 10)

      # * 4) tune_grid() ----
      boost_cube <- grid_latin_hypercube(
        tree_depth(),
        trees(),
        learn_rate(c(10^-5, 0.1), trans= NULL),
        finalize(mtry(), datB2_tbl),
        min_n(c(5, 40), trans= NULL),
        #loss_reduction(),
        #sample_size = sample_prop(),
        size = 30
      )

      set.seed(123)
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
        fit(syB ~ ., data = datB2_tbl)

    }
    if (modeling_method == "TREE") {
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
      set.seed(123)
      dat_folds <- vfold_cv(datB2_tbl,v = 10)

      # * 4) tune_grid() ----
      set.seed(123)
      rpart_cube <- grid_latin_hypercube(
        cost_complexity(),
        min_n(),
        tree_depth(c(3, 15), trans=NULL),
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
        fit(syB ~ ., data = datB2_tbl)

    }

  }
  iyA     <- predict(fit,datXA2)
  iyA     <- as.data.frame(iyA)
  iyA     <- as_tibble(iyA) %>% set_names(".pred")
  # Then, the median is taken of the column, not the iyA directly. That's where the
  # error was!

  # etheta1 and 2 are global mean and median (no domain)
  etheta1 <- sum(iyA$.pred*swA)/sum(swA)
  etheta2 <- mean(iyA$.pred)  #sum(iyA$.pred*weight)/sum(weight _LLCPWT)
  #etheta2 <- median(iyA$.pred)
  etheta  <- c(etheta1,etheta2)

  return(etheta)
}
