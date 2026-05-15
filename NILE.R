# =========================================================
# TFG - CAPÍTULO 4
# SERIE 4.4: NILE
# =========================================================

# =========================================================
# 0) PAQUETES
# =========================================================
req <- c("forecast", "keras3", "tensorflow", "Metrics", "ggplot2")
new <- setdiff(req, rownames(installed.packages()))
if(length(new)) install.packages(new, dependencies = TRUE)

library(forecast)
library(keras3)
library(tensorflow)
library(Metrics)
library(ggplot2)

# =========================================================
# 1) SEMILLA REPRODUCIBLE
# =========================================================
set.seed(123)
tensorflow::set_random_seed(123)

# =========================================================
# 2) FUNCIONES AUXILIARES GENERALES
# =========================================================

# -----------------------------
# 2.1 Escalado min-max
# -----------------------------
minmax_scale_train <- function(x){
  xmin <- min(x, na.rm = TRUE)
  xmax <- max(x, na.rm = TRUE)
  xs <- (x - xmin) / (xmax - xmin)
  list(x_s = as.numeric(xs), min = xmin, max = xmax)
}

minmax_scale_apply <- function(x, xmin, xmax){
  as.numeric((x - xmin) / (xmax - xmin))
}

minmax_invert <- function(x_s, xmin, xmax){
  as.numeric(x_s * (xmax - xmin) + xmin)
}

# -----------------------------
# 2.2 Features trigonométricas
# -----------------------------
make_seasonal_features <- function(y_ts){
  s <- frequency(y_ts)
  t <- seq_along(y_ts)
  sin_s <- sin(2 * pi * t / s)
  cos_s <- cos(2 * pi * t / s)
  cbind(sin_s = sin_s, cos_s = cos_s)
}

# -----------------------------
# 2.3 Construcción de ventanas
# -----------------------------
make_windows_multifeature <- function(y_scaled, season_mat, lookback){
  n <- length(y_scaled)
  if(n <= lookback) stop("No hay suficientes datos para crear ventanas.")
  
  X <- array(NA_real_, dim = c(n - lookback, lookback, 3))
  y <- numeric(n - lookback)
  
  for(i in seq_len(n - lookback)){
    idx <- i:(i + lookback - 1)
    X[i, , 1] <- y_scaled[idx]
    X[i, , 2] <- season_mat[idx, 1]
    X[i, , 3] <- season_mat[idx, 2]
    y[i] <- y_scaled[i + lookback]
  }
  
  list(X = X, y = y)
}

# -----------------------------
# 2.4 Split temporal train/validación
# -----------------------------
temporal_validation_split <- function(X, y, val_prop = 0.2){
  n <- dim(X)[1]
  n_val <- max(1, floor(n * val_prop))
  n_tr  <- n - n_val
  
  list(
    X_train = X[1:n_tr, , , drop = FALSE],
    y_train = y[1:n_tr],
    X_val   = X[(n_tr + 1):n, , , drop = FALSE],
    y_val   = y[(n_tr + 1):n]
  )
}

# -----------------------------
# 2.5 Modelo recurrente
# -----------------------------
build_rnn_model <- function(type = c("LSTM", "GRU"),
                            lookback,
                            n_features = 3,
                            units = 32,
                            dropout = 0.1,
                            lr = 0.001){
  
  type <- match.arg(type)
  
  inputs <- layer_input(shape = c(lookback, n_features))
  
  outputs <- if(type == "LSTM"){
    inputs %>%
      layer_lstm(
        units = units,
        dropout = dropout,
        recurrent_dropout = dropout
      ) %>%
      layer_dense(units = 1)
  } else {
    inputs %>%
      layer_gru(
        units = units,
        dropout = dropout,
        recurrent_dropout = dropout
      ) %>%
      layer_dense(units = 1)
  }
  
  model <- keras_model(inputs = inputs, outputs = outputs)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss = "mse"
  )
  
  model
}

# -----------------------------
# 2.6 Forecast recursivo multi-step
# -----------------------------
recursive_forecast_multifeature <- function(model,
                                            y_hist_scaled,
                                            season_hist,
                                            season_future,
                                            h,
                                            lookback){
  
  preds <- numeric(h)
  y_buf <- as.numeric(tail(y_hist_scaled, lookback))
  seas_all <- rbind(season_hist, season_future)
  
  for(i in 1:h){
    start_idx <- nrow(season_hist) + i - lookback
    end_idx   <- nrow(season_hist) + i - 1
    
    sin_buf <- seas_all[start_idx:end_idx, 1]
    cos_buf <- seas_all[start_idx:end_idx, 2]
    
    X_in <- array(NA_real_, dim = c(1, lookback, 3))
    X_in[1, , 1] <- y_buf
    X_in[1, , 2] <- sin_buf
    X_in[1, , 3] <- cos_buf
    
    p <- as.numeric(predict(model, X_in, verbose = 0))
    preds[i] <- p
    
    y_buf <- c(y_buf[-1], p)
  }
  
  preds
}

# -----------------------------
# 2.7 Entrenamiento DL
# -----------------------------
fit_rnn_univariate_improved <- function(y_train_ts,
                                        h,
                                        lookback = 4,
                                        type = c("LSTM", "GRU"),
                                        units = 32,
                                        dropout = 0.1,
                                        batch_size = 8,
                                        epochs = 200,
                                        lr = 0.001,
                                        val_prop = 0.2,
                                        patience = 15,
                                        verbose_fit = 0){
  
  type <- match.arg(type)
  
  sc <- minmax_scale_train(y_train_ts)
  y_scaled <- sc$x_s
  
  season_train <- make_seasonal_features(y_train_ts)
  
  w <- make_windows_multifeature(y_scaled, season_train, lookback = lookback)
  split <- temporal_validation_split(w$X, w$y, val_prop = val_prop)
  
  model <- build_rnn_model(
    type = type,
    lookback = lookback,
    n_features = 3,
    units = units,
    dropout = dropout,
    lr = lr
  )
  
  cb_es <- callback_early_stopping(
    monitor = "val_loss",
    patience = patience,
    restore_best_weights = TRUE
  )
  
  cb_rlr <- callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = max(3, floor(patience / 3)),
    min_lr = 1e-5
  )
  
  history <- model %>% fit(
    x = split$X_train,
    y = split$y_train,
    validation_data = list(split$X_val, split$y_val),
    epochs = epochs,
    batch_size = batch_size,
    shuffle = FALSE,
    callbacks = list(cb_es, cb_rlr),
    verbose = verbose_fit
  )
  
  s <- frequency(y_train_ts)
  t_hist <- seq_along(y_train_ts)
  t_future <- (length(t_hist) + 1):(length(t_hist) + h)
  
  season_future <- cbind(
    sin_s = sin(2 * pi * t_future / s),
    cos_s = cos(2 * pi * t_future / s)
  )
  
  preds_scaled <- recursive_forecast_multifeature(
    model = model,
    y_hist_scaled = y_scaled,
    season_hist = season_train,
    season_future = season_future,
    h = h,
    lookback = lookback
  )
  
  preds <- minmax_invert(preds_scaled, sc$min, sc$max)
  
  fc_ts <- ts(
    preds,
    start = tsp(y_train_ts)[2] + 1 / frequency(y_train_ts),
    frequency = frequency(y_train_ts)
  )
  
  val_pred_scaled <- as.numeric(predict(model, split$X_val, verbose = 0))
  val_pred <- minmax_invert(val_pred_scaled, sc$min, sc$max)
  val_real <- minmax_invert(split$y_val, sc$min, sc$max)
  
  val_rmse <- sqrt(mean((val_real - val_pred)^2))
  val_mae  <- mean(abs(val_real - val_pred))
  val_mape <- mean(abs((val_real - val_pred) / val_real)) * 100
  
  list(
    model = model,
    history = history,
    fc = fc_ts,
    val_rmse = val_rmse,
    val_mae = val_mae,
    val_mape = val_mape,
    params = list(
      type = type,
      lookback = lookback,
      units = units,
      dropout = dropout,
      batch_size = batch_size,
      epochs = epochs,
      lr = lr,
      val_prop = val_prop,
      patience = patience
    )
  )
}

# -----------------------------
# 2.8 Forecast object DL
# -----------------------------
make_dl_forecast_object <- function(y_train, y_pred){
  fc <- forecast::forecast(y_train, h = length(y_pred))
  fc$mean <- y_pred
  fc
}

# -----------------------------
# 2.9 Métricas
# -----------------------------
calc_metrics <- function(actual, pred){
  data.frame(
    RMSE = sqrt(mean((actual - pred)^2)),
    MAE  = mean(abs(actual - pred)),
    MAPE = mean(abs((actual - pred) / actual)) * 100
  )
}

# -----------------------------
# 2.10 Ranking
# -----------------------------
rank_models_test <- function(models_list, y_test_ts){
  out <- lapply(names(models_list), function(nm){
    pred <- as.numeric(models_list[[nm]]$mean)
    act  <- as.numeric(y_test_ts)
    data.frame(
      Model = nm,
      RMSE = sqrt(mean((act - pred)^2)),
      MAE  = mean(abs(act - pred)),
      MAPE = mean(abs((act - pred) / act)) * 100
    )
  })
  out <- do.call(rbind, out)
  out[order(out$RMSE, out$MAE), ]
}

# -----------------------------
# 2.11 Diebold-Mariano
# -----------------------------
dm_compare <- function(actual, pred1, pred2, h = 1, power = 2){
  e1 <- actual - pred1
  e2 <- actual - pred2
  forecast::dm.test(e1, e2, h = h, power = power)
}

# -----------------------------
# 2.12 LSTM sobre residuos
# -----------------------------
fit_lstm_residuals <- function(resid_train_ts,
                               h,
                               lookback = 8,
                               units = 64,
                               dropout = 0,
                               batch_size = 8,
                               epochs = 200,
                               lr = 0.001,
                               val_prop = 0.2,
                               patience = 15,
                               verbose_fit = 0){
  
  sc <- minmax_scale_train(resid_train_ts)
  y_scaled <- sc$x_s
  
  season_train <- make_seasonal_features(resid_train_ts)
  w <- make_windows_multifeature(y_scaled, season_train, lookback = lookback)
  split <- temporal_validation_split(w$X, w$y, val_prop = val_prop)
  
  model <- build_rnn_model(
    type = "LSTM",
    lookback = lookback,
    n_features = 3,
    units = units,
    dropout = dropout,
    lr = lr
  )
  
  cb_es <- callback_early_stopping(
    monitor = "val_loss",
    patience = patience,
    restore_best_weights = TRUE
  )
  
  cb_rlr <- callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = max(3, floor(patience / 3)),
    min_lr = 1e-5
  )
  
  history <- model %>% fit(
    x = split$X_train,
    y = split$y_train,
    validation_data = list(split$X_val, split$y_val),
    epochs = epochs,
    batch_size = batch_size,
    shuffle = FALSE,
    callbacks = list(cb_es, cb_rlr),
    verbose = verbose_fit
  )
  
  s <- frequency(resid_train_ts)
  t_hist <- seq_along(resid_train_ts)
  t_future <- (length(t_hist) + 1):(length(t_hist) + h)
  
  season_future <- cbind(
    sin_s = sin(2 * pi * t_future / s),
    cos_s = cos(2 * pi * t_future / s)
  )
  
  resid_pred_scaled <- recursive_forecast_multifeature(
    model = model,
    y_hist_scaled = y_scaled,
    season_hist = season_train,
    season_future = season_future,
    h = h,
    lookback = lookback
  )
  
  resid_pred <- minmax_invert(resid_pred_scaled, sc$min, sc$max)
  
  resid_fc_ts <- ts(
    resid_pred,
    start = tsp(resid_train_ts)[2] + 1 / frequency(resid_train_ts),
    frequency = frequency(resid_train_ts)
  )
  
  list(
    model = model,
    history = history,
    resid_fc = resid_fc_ts
  )
}

# -----------------------------
# 2.13 Forecast object genérico
# -----------------------------
make_forecast_object <- function(y_train, y_pred){
  fc <- forecast::forecast(y_train, h = length(y_pred))
  fc$mean <- y_pred
  fc
}

# =========================================================
# 3) 4.4.1 DESCRIPCIÓN DE LA SERIE
# =========================================================
y0 <- Nile

cat("\n=====================================================\n")
cat("4.4.1 DESCRIPCIÓN DE LA SERIE: NILE\n")
cat("=====================================================\n")
cat("Frecuencia:", frequency(y0), "\n")
cat("Inicio:", paste(start(y0), collapse = "-"), "\n")
cat("Fin:", paste(end(y0), collapse = "-"), "\n")
cat("Número de observaciones:", length(y0), "\n")

# =========================================================
# 4) 4.4.2 ANÁLISIS EXPLORATORIO Y TRANSFORMACIONES
# =========================================================
plot(
  y0,
  main = "Nile - Serie original",
  ylab = "Caudal anual",
  col = "black",
  lwd = 2
)

# Transformación logarítmica
y <- log(y0)

plot(
  y,
  main = "Nile - Serie en logaritmos",
  ylab = "log(Caudal anual)",
  col = "blue",
  lwd = 2
)

acf(y, main = "ACF de log(Nile)")
pacf(y, main = "PACF de log(Nile)")

dy <- diff(y, differences = 1)

plot(
  dy,
  main = "Primera diferencia de log(Nile)",
  ylab = "Diferencia",
  col = "darkgreen"
)

acf(dy, main = "ACF de la primera diferencia de log(Nile)")
pacf(dy, main = "PACF de la primera diferencia de log(Nile)")

# =========================================================
# 5) 4.4.3 MODELIZACIÓN CLÁSICA
# =========================================================
h <- 12
n <- length(y)

y_train <- window(y, end = time(y)[n - h])
y_test  <- window(y, start = time(y)[n - h + 1])

cat("\n=====================================================\n")
cat("DIVISIÓN TRAIN / TEST\n")
cat("=====================================================\n")
cat("Observaciones train:", length(y_train), "\n")
cat("Observaciones test:", length(y_test), "\n")
cat("Horizonte de predicción h:", h, "\n")

fc_naive <- naive(y_train, h = h)
fc_drift <- rwf(y_train, h = h, drift = TRUE)

fit_trend <- tslm(y_train ~ trend)
fc_trend  <- forecast(fit_trend, h = h)

fit_arima <- Arima(
  y_train,
  order = c(1, 1, 1)
)
fc_arima <- forecast(fit_arima, h = h)

fit_auto <- auto.arima(
  y_train,
  seasonal = FALSE,
  stepwise = FALSE,
  approximation = FALSE
)
fc_auto <- forecast(fit_auto, h = h)

fit_ets <- ets(y_train)
fc_ets  <- forecast(fit_ets, h = h)

cat("\n================ RESUMEN ARIMA MANUAL ================\n")
print(summary(fit_arima))
cat("AIC ARIMA:", AIC(fit_arima), "\n")
cat("BIC ARIMA:", BIC(fit_arima), "\n")

cat("\n================ RESUMEN AUTOARIMA ================\n")
print(summary(fit_auto))
cat("AIC AutoARIMA:", AIC(fit_auto), "\n")
cat("BIC AutoARIMA:", BIC(fit_auto), "\n")

res_arima <- residuals(fit_arima)

ts.plot(
  res_arima,
  main = "Residuos del modelo ARIMA",
  ylab = "Residuo",
  col = "darkred"
)
abline(h = 0, lty = 2)

acf(res_arima, main = "ACF residuos ARIMA")
pacf(res_arima, main = "PACF residuos ARIMA")

cat("\n================ LJUNG-BOX ARIMA ================\n")
print(Box.test(res_arima, lag = 8, type = "Ljung-Box"))

models_classic <- list(
  Naive = fc_naive,
  Drift = fc_drift,
  Trend = fc_trend,
  ARIMA = fc_arima,
  AutoARIMA = fc_auto,
  ETS = fc_ets
)

ranking_classic <- rank_models_test(models_classic, y_test)

cat("\n================ RANKING CLÁSICO EN TEST ================\n")
print(ranking_classic)

ranking_classic$Model <- factor(
  ranking_classic$Model,
  levels = ranking_classic$Model[order(ranking_classic$RMSE)]
)

p_classic <- ggplot(ranking_classic, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  geom_text(aes(label = round(RMSE, 4)),
            hjust = -0.2,
            size = 4) +
  labs(
    title = "Comparación de modelos clásicos - Nile (Test, log)",
    x = "Modelo",
    y = "RMSE"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(face = "bold")
  ) +
  ylim(0, max(ranking_classic$RMSE) * 1.1)

print(p_classic)

df_plot_arima <- data.frame(
  time = as.numeric(time(y_train)),
  Real = as.numeric(y_train),
  Ajuste_ARIMA = as.numeric(fitted(fit_arima))
)

p_arima <- ggplot(df_plot_arima, aes(x = time)) +
  geom_line(aes(y = Real, colour = "Serie real"), linewidth = 1) +
  geom_line(aes(y = Ajuste_ARIMA, colour = "Ajuste ARIMA"),
            linewidth = 1,
            linetype = "dashed") +
  scale_colour_manual(
    values = c("Serie real" = "black", "Ajuste ARIMA" = "red")
  ) +
  labs(
    title = "Ajuste del modelo ARIMA sobre log(Nile)",
    x = "Tiempo",
    y = "log(Caudal anual)",
    colour = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

print(p_arima)

# =========================================================
# 6) 4.4.4 MODELIZACIÓN MEDIANTE DEEP LEARNING
# =========================================================
grid <- expand.grid(
  type = c("LSTM", "GRU"),
  lookback = c(4, 8, 12),
  units = c(16, 32, 64),
  dropout = c(0.0, 0.1),
  batch_size = c(4, 8),
  stringsAsFactors = FALSE
)

results <- list()
fits <- list()

for(i in seq_len(nrow(grid))){
  cat("\n====================================\n")
  cat("Configuración", i, "de", nrow(grid), "\n")
  print(grid[i, ])
  
  fit_i <- fit_rnn_univariate_improved(
    y_train_ts = y_train,
    h = h,
    lookback = grid$lookback[i],
    type = grid$type[i],
    units = grid$units[i],
    dropout = grid$dropout[i],
    batch_size = grid$batch_size[i],
    epochs = 200,
    lr = 0.001,
    val_prop = 0.2,
    patience = 15,
    verbose_fit = 0
  )
  
  test_metrics <- calc_metrics(
    actual = as.numeric(y_test),
    pred   = as.numeric(fit_i$fc)
  )
  
  row_i <- cbind(
    grid[i, ],
    val_rmse = fit_i$val_rmse,
    val_mae  = fit_i$val_mae,
    val_mape = fit_i$val_mape,
    test_metrics
  )
  
  results[[i]] <- row_i
  fits[[i]] <- fit_i
}

results_df <- do.call(rbind, results)
results_df <- results_df[order(results_df$val_rmse, results_df$val_mae), ]
row.names(results_df) <- NULL

cat("\n================= RANKING DL POR VALIDACIÓN =================\n")
print(results_df)

best_row <- results_df[1, ]
cat("\n============= MEJOR CONFIGURACIÓN DL (POR VALIDACIÓN) =============\n")
print(best_row)

best_idx <- which(
  grid$type == best_row$type &
    grid$lookback == best_row$lookback &
    grid$units == best_row$units &
    grid$dropout == best_row$dropout &
    grid$batch_size == best_row$batch_size
)[1]

best_fit <- fits[[best_idx]]

best_dl_name <- paste0(best_row$type, "_BEST")
best_dl_fc <- make_dl_forecast_object(y_train, best_fit$fc)

models_nile <- models_classic
models_nile[[best_dl_name]] <- best_dl_fc

ranking_final <- rank_models_test(models_nile, y_test)

cat("\n================ RANKING FINAL EN TEST =================\n")
print(ranking_final)

best_dl_label <- as.character(best_row$type)

df_plot_dl_real <- data.frame(
  time = as.numeric(time(y)),
  Real = as.numeric(y)
)

df_plot_dl_pred <- data.frame(
  time = as.numeric(time(y_test)),
  Pred_DL = as.numeric(best_fit$fc)
)

p_dl <- ggplot() +
  geom_line(data = df_plot_dl_real,
            aes(x = time, y = Real, colour = "Serie real"),
            linewidth = 1) +
  geom_line(data = df_plot_dl_pred,
            aes(x = time, y = Pred_DL, colour = best_dl_label),
            linewidth = 1) +
  scale_colour_manual(
    values = setNames(c("black", "blue"), c("Serie real", best_dl_label))
  ) +
  labs(
    title = "Predicción del mejor modelo DL sobre log(Nile)",
    x = "Tiempo",
    y = "log(Caudal anual)",
    colour = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

print(p_dl)

# =========================================================
# 7) 4.4.5 COMPARACIÓN DE RESULTADOS PREDICTIVOS
# =========================================================
cat("\n=====================================================\n")
cat("COMPARACIÓN PRINCIPAL DE MODELOS\n")
cat("=====================================================\n")
print(ranking_final)

ranking_final$Model <- factor(
  ranking_final$Model,
  levels = ranking_final$Model[order(ranking_final$RMSE)]
)

ranking_final$Tipo <- ifelse(
  ranking_final$Model %in% c("ARIMA", "AutoARIMA"),
  "Clásico (Mejor)",
  ifelse(
    ranking_final$Model == best_dl_name,
    "Deep Learning",
    "Otros"
  )
)

p_final <- ggplot(ranking_final, aes(x = Model, y = RMSE, fill = Tipo)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  geom_text(aes(label = round(RMSE, 4)),
            hjust = -0.2,
            size = 4) +
  scale_fill_manual(
    values = c(
      "Clásico (Mejor)" = "black",
      "Deep Learning" = "blue",
      "Otros" = "grey70"
    )
  ) +
  labs(
    title = "Comparación final de modelos - Nile (Test, log)",
    x = "Modelo",
    y = "RMSE",
    fill = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  ) +
  ylim(0, max(ranking_final$RMSE) * 1.1)

print(p_final)

# =========================================================
# 8) 4.4.6 ANÁLISIS DE ESTABILIDAD DEL MEJOR MODELO DL
# =========================================================
best_type <- as.character(best_row$type)
best_lookback <- best_row$lookback
best_units <- best_row$units
best_dropout <- best_row$dropout
best_batch_size <- best_row$batch_size

n_reps <- 10
seeds <- 1001:(1000 + n_reps)

rep_results <- data.frame(
  rep = integer(),
  seed = integer(),
  RMSE = numeric(),
  MAE = numeric(),
  MAPE = numeric(),
  stringsAsFactors = FALSE
)

rep_forecasts <- vector("list", n_reps)

for(i in seq_len(n_reps)){
  
  cat("\n====================================\n")
  cat("Repetición", i, "de", n_reps, "- seed =", seeds[i], "\n")
  
  set.seed(seeds[i])
  tensorflow::set_random_seed(seeds[i])
  
  fit_i <- fit_rnn_univariate_improved(
    y_train_ts = y_train,
    h = h,
    lookback = best_lookback,
    type = best_type,
    units = best_units,
    dropout = best_dropout,
    batch_size = best_batch_size,
    epochs = 200,
    lr = 0.001,
    val_prop = 0.2,
    patience = 15,
    verbose_fit = 0
  )
  
  met_i <- calc_metrics(
    actual = as.numeric(y_test),
    pred   = as.numeric(fit_i$fc)
  )
  
  rep_results <- rbind(
    rep_results,
    data.frame(
      rep = i,
      seed = seeds[i],
      RMSE = met_i$RMSE,
      MAE  = met_i$MAE,
      MAPE = met_i$MAPE
    )
  )
  
  rep_forecasts[[i]] <- fit_i$fc
}

summary_rep <- data.frame(
  Metric = c("RMSE", "MAE", "MAPE"),
  Mean = c(mean(rep_results$RMSE),
           mean(rep_results$MAE),
           mean(rep_results$MAPE)),
  SD = c(sd(rep_results$RMSE),
         sd(rep_results$MAE),
         sd(rep_results$MAPE)),
  Min = c(min(rep_results$RMSE),
          min(rep_results$MAE),
          min(rep_results$MAPE)),
  Max = c(max(rep_results$RMSE),
          max(rep_results$MAE),
          max(rep_results$MAPE))
)

cat("\n================ RESULTADOS POR REPETICIÓN ================\n")
print(rep_results)

cat("\n================ RESUMEN ESTADÍSTICO ================\n")
print(summary_rep)

best_rep_idx <- which.min(rep_results$RMSE)
worst_rep_idx <- which.max(rep_results$RMSE)

cat("\nMejor repetición:\n")
print(rep_results[best_rep_idx, ])

cat("\nPeor repetición:\n")
print(rep_results[worst_rep_idx, ])

plot(
  rep_results$rep, rep_results$RMSE,
  type = "b", pch = 19,
  xlab = "Repetición",
  ylab = "RMSE en test",
  main = "Estabilidad de la mejor configuración DL"
)
abline(h = mean(rep_results$RMSE), lty = 2)

ts.plot(
  y,
  col = "black",
  lwd = 2,
  ylab = "log(Caudal anual)",
  main = paste0("log(Nile) - Repeticiones del mejor ", best_type)
)

for(i in seq_len(n_reps)){
  lines(rep_forecasts[[i]], lwd = 1, lty = 1)
}

lines(rep_forecasts[[best_rep_idx]], lwd = 3, col = "blue")

legend(
  "topleft",
  legend = c("Serie real", "Repeticiones DL", "Mejor repetición"),
  col = c("black", "gray40", "blue"),
  lty = c(1, 1, 1),
  lwd = c(2, 1, 3),
  bty = "n"
)

# =========================================================
# 9) MODELO HÍBRIDO ARIMA + LSTM SOBRE RESIDUOS
# =========================================================
resid_train <- residuals(fit_arima)
resid_train <- ts(
  as.numeric(resid_train),
  start = start(y_train),
  frequency = frequency(y_train)
)

cat("\n================ RESUMEN RESIDUOS ARIMA ================\n")
print(summary(as.numeric(resid_train)))

fit_res_lstm <- fit_lstm_residuals(
  resid_train_ts = resid_train,
  h = h,
  lookback = 8,
  units = 64,
  dropout = 0,
  batch_size = 8,
  epochs = 200,
  lr = 0.001,
  val_prop = 0.2,
  patience = 15,
  verbose_fit = 0
)

fc_resid_lstm <- fit_res_lstm$resid_fc

fc_hybrid_mean <- fc_arima$mean + fc_resid_lstm
fc_hybrid <- make_forecast_object(y_train, fc_hybrid_mean)

metrics_arima  <- calc_metrics(as.numeric(y_test), as.numeric(fc_arima$mean))
metrics_hybrid <- calc_metrics(as.numeric(y_test), as.numeric(fc_hybrid$mean))

comparison_hybrid <- rbind(
  data.frame(Model = "ARIMA", metrics_arima),
  data.frame(Model = "ARIMA_LSTM_HYBRID", metrics_hybrid)
)

cat("\n================ COMPARACIÓN HÍBRIDO VS ARIMA ================\n")
print(comparison_hybrid)

models_all <- models_nile
models_all[["ARIMA_LSTM_HYBRID"]] <- fc_hybrid

ranking_hybrid <- rank_models_test(models_all, y_test)

cat("\n================ RANKING FINAL CON HÍBRIDO ================\n")
print(ranking_hybrid)

ranking_hybrid$Model <- factor(
  ranking_hybrid$Model,
  levels = ranking_hybrid$Model[order(ranking_hybrid$RMSE)]
)

ranking_hybrid$Tipo <- ifelse(
  ranking_hybrid$Model %in% c("ARIMA", "AutoARIMA"),
  "Clásico (Mejor)",
  ifelse(
    ranking_hybrid$Model == best_dl_name,
    "Deep Learning",
    ifelse(
      ranking_hybrid$Model == "ARIMA_LSTM_HYBRID",
      "Híbrido",
      "Otros"
    )
  )
)

p_hybrid <- ggplot(ranking_hybrid, aes(x = Model, y = RMSE, fill = Tipo)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  geom_text(aes(label = round(RMSE, 4)),
            hjust = -0.2,
            size = 4) +
  scale_fill_manual(
    values = c(
      "Clásico (Mejor)" = "black",
      "Deep Learning" = "blue",
      "Híbrido" = "darkgreen",
      "Otros" = "grey70"
    )
  ) +
  labs(
    title = "Comparación final de modelos (incluyendo híbrido) - Nile",
    x = "Modelo",
    y = "RMSE",
    fill = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  ) +
  ylim(0, max(ranking_hybrid$RMSE) * 1.1)

print(p_hybrid)

ts.plot(
  y,
  col = "black",
  lwd = 2,
  ylab = "log(Caudal anual)",
  main = "log(Nile): ARIMA vs ARIMA+LSTM híbrido"
)

lines(fc_arima$mean, col = "red", lwd = 2, lty = 2)
lines(fc_hybrid$mean, col = "blue", lwd = 2)

legend(
  "topleft",
  legend = c("Serie real", "ARIMA", "ARIMA + LSTM"),
  col = c("black", "red", "blue"),
  lty = c(1, 2, 1),
  lwd = 2,
  bty = "n"
)

plot(
  resid_train,
  main = "Residuos del modelo ARIMA",
  ylab = "Residuo",
  col = "darkred"
)
abline(h = 0, lty = 2)

# =========================================================
# 10) CONTRASTE ESTADÍSTICO DE PREDICCIONES
# =========================================================
dm_arima_bestdl <- dm_compare(
  actual = as.numeric(y_test),
  pred1 = as.numeric(fc_arima$mean),
  pred2 = as.numeric(best_dl_fc$mean),
  h = 1,
  power = 2
)

dm_arima_ets <- dm_compare(
  actual = as.numeric(y_test),
  pred1 = as.numeric(fc_arima$mean),
  pred2 = as.numeric(fc_ets$mean),
  h = 1,
  power = 2
)

dm_arima_hybrid <- dm_compare(
  actual = as.numeric(y_test),
  pred1 = as.numeric(fc_arima$mean),
  pred2 = as.numeric(fc_hybrid$mean),
  h = 1,
  power = 2
)

cat("\n================ DIEBOLD-MARIANO =================\n")
cat("\nARIMA vs BEST_DL\n")
print(dm_arima_bestdl)

cat("\nARIMA vs ETS\n")
print(dm_arima_ets)

cat("\nARIMA vs HYBRID\n")
print(dm_arima_hybrid)

# =========================================================
# 11) EXPORTACIÓN DE RESULTADOS
# =========================================================
write.csv(ranking_classic, "nile_ranking_classic.csv", row.names = FALSE)
write.csv(results_df, "nile_dl_grid_results.csv", row.names = FALSE)
write.csv(ranking_final, "nile_final_ranking.csv", row.names = FALSE)
write.csv(rep_results, "nile_lstm_repeticiones.csv", row.names = FALSE)
write.csv(summary_rep, "nile_lstm_repeticiones_resumen.csv", row.names = FALSE)
write.csv(comparison_hybrid, "nile_arima_vs_hybrid.csv", row.names = FALSE)
write.csv(ranking_hybrid, "nile_hybrid_ranking.csv", row.names = FALSE)

# =========================================================
# FIN DEL SCRIPT
# =========================================================