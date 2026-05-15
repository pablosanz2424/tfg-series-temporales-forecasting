# =========================================================
# TFG - CAPÍTULO 4
# SERIE 4.8: TIPO DE CAMBIO USD/EUR
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
# 2.2 Features estacionales trigonométricas
# -----------------------------
make_seasonal_features <- function(y_ts){
  s <- frequency(y_ts)
  t <- seq_along(y_ts)
  sin_s <- sin(2 * pi * t / s)
  cos_s <- cos(2 * pi * t / s)
  cbind(sin_s = sin_s, cos_s = cos_s)
}

# -----------------------------
# 2.3 Construcción de ventanas multivariantes
# feature 1 = serie escalada
# feature 2 = sin estacional
# feature 3 = cos estacional
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
# 2.4 División temporal train/validación
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
# 2.5 Métricas
# -----------------------------
calc_metrics <- function(actual, pred){
  data.frame(
    RMSE = Metrics::rmse(actual, pred),
    MAE  = Metrics::mae(actual, pred),
    MAPE = mean(abs((actual - pred) / actual)) * 100
  )
}

# -----------------------------
# 2.6 Ranking de modelos sobre test
# -----------------------------
rank_models_test <- function(models_list, y_test){
  out <- lapply(names(models_list), function(nm){
    met <- calc_metrics(
      actual = as.numeric(y_test),
      pred   = as.numeric(models_list[[nm]]$mean)
    )
    data.frame(Model = nm, met)
  })
  out <- do.call(rbind, out)
  out <- out[order(out$RMSE, out$MAE), ]
  rownames(out) <- NULL
  out
}

# -----------------------------
# 2.7 Constructor RNN
# -----------------------------
build_rnn_model <- function(input_shape, type = c("LSTM", "GRU"),
                            units = 32, dropout = 0, lr = 0.001){
  
  type <- match.arg(type)
  
  model <- keras_model_sequential()
  
  if(type == "LSTM"){
    model <- model |>
      layer_lstm(
        units = units,
        input_shape = input_shape,
        dropout = dropout,
        recurrent_dropout = 0
      )
  } else {
    model <- model |>
      layer_gru(
        units = units,
        input_shape = input_shape,
        dropout = dropout,
        recurrent_dropout = 0
      )
  }
  
  model <- model |>
    layer_dense(units = 1)
  
  model |>
    compile(
      loss = "mse",
      optimizer = optimizer_adam(learning_rate = lr),
      metrics = list("mae")
    )
  
  model
}

# -----------------------------
# 2.8 Predicción recursiva multifeature
# -----------------------------
recursive_forecast_multifeature <- function(model,
                                            y_hist_scaled,
                                            season_hist,
                                            season_future,
                                            h,
                                            lookback){
  preds <- numeric(h)
  y_ext <- as.numeric(y_hist_scaled)
  season_ext <- season_hist
  
  for(i in 1:h){
    idx <- (length(y_ext) - lookback + 1):length(y_ext)
    
    x_new <- array(NA_real_, dim = c(1, lookback, 3))
    x_new[1, , 1] <- y_ext[idx]
    x_new[1, , 2] <- season_ext[idx, 1]
    x_new[1, , 3] <- season_ext[idx, 2]
    
    pred_i <- as.numeric(predict(model, x_new, verbose = 0))
    preds[i] <- pred_i
    
    y_ext <- c(y_ext, pred_i)
    season_ext <- rbind(season_ext, season_future[i, , drop = FALSE])
  }
  
  preds
}

# -----------------------------
# 2.9 Ajuste RNN univariante mejorado
# -----------------------------
fit_rnn_univariate_improved <- function(y_train_ts,
                                        h,
                                        lookback,
                                        type = c("LSTM", "GRU"),
                                        units = 32,
                                        dropout = 0,
                                        batch_size = 8,
                                        epochs = 200,
                                        lr = 0.001,
                                        val_prop = 0.2,
                                        patience = 15,
                                        verbose_fit = 0){
  
  type <- match.arg(type)
  
  sc <- minmax_scale_train(as.numeric(y_train_ts))
  y_scaled <- sc$x_s
  
  season_train <- make_seasonal_features(y_train_ts)
  windows <- make_windows_multifeature(y_scaled, season_train, lookback)
  split <- temporal_validation_split(windows$X, windows$y, val_prop = val_prop)
  
  input_shape <- c(dim(split$X_train)[2], dim(split$X_train)[3])
  
  model <- build_rnn_model(
    input_shape = input_shape,
    type = type,
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
    patience = 5,
    min_lr = 1e-5
  )
  
  history <- model |>
    fit(
      x = split$X_train,
      y = split$y_train,
      validation_data = list(split$X_val, split$y_val),
      epochs = epochs,
      batch_size = batch_size,
      shuffle = FALSE,
      callbacks = list(cb_es, cb_rlr),
      verbose = verbose_fit
    )
  
  val_pred_scaled <- as.numeric(predict(model, split$X_val, verbose = 0))
  val_pred <- minmax_invert(val_pred_scaled, sc$min, sc$max)
  val_real <- minmax_invert(split$y_val, sc$min, sc$max)
  val_metrics <- calc_metrics(val_real, val_pred)
  
  s <- frequency(y_train_ts)
  t_hist <- seq_along(y_train_ts)
  t_future <- (length(t_hist) + 1):(length(t_hist) + h)
  
  season_future <- cbind(
    sin_s = sin(2 * pi * t_future / s),
    cos_s = cos(2 * pi * t_future / s)
  )
  
  fc_scaled <- recursive_forecast_multifeature(
    model = model,
    y_hist_scaled = y_scaled,
    season_hist = season_train,
    season_future = season_future,
    h = h,
    lookback = lookback
  )
  
  fc <- minmax_invert(fc_scaled, sc$min, sc$max)
  fc_ts <- ts(
    fc,
    start = tsp(y_train_ts)[2] + 1 / frequency(y_train_ts),
    frequency = frequency(y_train_ts)
  )
  
  list(
    model = model,
    history = history,
    fc = fc_ts,
    val_rmse = val_metrics$RMSE,
    val_mae  = val_metrics$MAE,
    val_mape = val_metrics$MAPE
  )
}

# -----------------------------
# 2.10 Repeticiones del mejor DL
# -----------------------------
repeat_best_dl <- function(y_train, y_test, best_row, n_reps = 10){
  out <- list()
  
  for(i in 1:n_reps){
    set.seed(100 + i)
    tensorflow::set_random_seed(100 + i)
    
    fit_i <- fit_rnn_univariate_improved(
      y_train_ts = y_train,
      h = length(y_test),
      lookback = best_row$lookback,
      type = as.character(best_row$type),
      units = best_row$units,
      dropout = best_row$dropout,
      batch_size = best_row$batch_size,
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
    
    out[[i]] <- data.frame(
      Repetition = i,
      ModelType = as.character(best_row$type),
      RMSE = met_i$RMSE,
      MAE  = met_i$MAE,
      MAPE = met_i$MAPE
    )
  }
  
  rep_df <- do.call(rbind, out)
  rownames(rep_df) <- NULL
  rep_df
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
# 2.12 Híbrido SARIMA + LSTM sobre residuos
# -----------------------------
fit_hybrid_residual_lstm <- function(fit_sarima, y_train_ts, h,
                                     lookback = 12,
                                     units = 32,
                                     dropout = 0,
                                     batch_size = 8,
                                     epochs = 200,
                                     lr = 0.001){
  
  resid_train <- residuals(fit_sarima)
  resid_train <- as.numeric(resid_train)
  
  resid_train_ts <- ts(
    resid_train,
    start = start(y_train_ts),
    frequency = frequency(y_train_ts)
  )
  
  sc <- minmax_scale_train(resid_train)
  y_scaled <- sc$x_s
  
  season_train <- make_seasonal_features(resid_train_ts)
  windows <- make_windows_multifeature(y_scaled, season_train, lookback)
  split <- temporal_validation_split(windows$X, windows$y, val_prop = 0.2)
  
  input_shape <- c(dim(split$X_train)[2], dim(split$X_train)[3])
  
  model <- build_rnn_model(
    input_shape = input_shape,
    type = "LSTM",
    units = units,
    dropout = dropout,
    lr = lr
  )
  
  cb_es <- callback_early_stopping(
    monitor = "val_loss",
    patience = 15,
    restore_best_weights = TRUE
  )
  
  cb_rlr <- callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 5,
    min_lr = 1e-5
  )
  
  history <- model |>
    fit(
      x = split$X_train,
      y = split$y_train,
      validation_data = list(split$X_val, split$y_val),
      epochs = epochs,
      batch_size = batch_size,
      shuffle = FALSE,
      callbacks = list(cb_es, cb_rlr),
      verbose = 0
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
# 3) 4.8.1 DESCRIPCIÓN DE LA SERIE
# =========================================================
df <- TIPO_DE_CAMBIO_DOLAR_EURO
df$observation_date <- as.Date(df$observation_date)
df$DEXUSEU <- as.numeric(df$DEXUSEU)
df <- df[order(df$observation_date), ]
df <- na.omit(df)

y0 <- ts(df$DEXUSEU, start = c(2000, 1), frequency = 12)

cat("\n=====================================================\n")
cat("4.8.1 DESCRIPCIÓN DE LA SERIE: TIPO DE CAMBIO USD/EUR\n")
cat("=====================================================\n")
cat("Frecuencia:", frequency(y0), "\n")
cat("Inicio:", paste(start(y0), collapse = "-"), "\n")
cat("Fin:", paste(end(y0), collapse = "-"), "\n")
cat("Número de observaciones:", length(y0), "\n")

# =========================================================
# 4) 4.8.2 ANÁLISIS EXPLORATORIO Y TRANSFORMACIONES
# =========================================================
plot(
  y0,
  main = "Tipo de cambio USD/EUR - Serie original",
  ylab = "USD por EUR",
  col = "black",
  lwd = 2
)

y <- log(y0)

plot(
  y,
  main = "Tipo de cambio USD/EUR - Serie en logaritmos",
  ylab = "log(USD/EUR)",
  col = "blue",
  lwd = 2
)

forecast::ggseasonplot(y0, year.labels = TRUE, year.labels.left = TRUE) +
  ylab("USD por EUR") +
  ggtitle("Tipo de cambio USD/EUR - Seasonal plot (serie original)")

forecast::ggseasonplot(y, year.labels = TRUE, year.labels.left = TRUE) +
  ylab("log(USD/EUR)") +
  ggtitle("Tipo de cambio USD/EUR - Seasonal plot (log)")

forecast::ggsubseriesplot(y0) +
  ylab("USD por EUR") +
  ggtitle("Tipo de cambio USD/EUR - Subseries plot")

acf(y, main = "ACF de log(USD/EUR)")
pacf(y, main = "PACF de log(USD/EUR)")

dy  <- diff(y, differences = 1)
dsy <- diff(y, lag = frequency(y), differences = 1)

plot(dy,  main = "Primera diferencia de log(USD/EUR)", ylab = "Diferencia")
plot(dsy, main = "Diferencia estacional de log(USD/EUR)", ylab = "Diferencia estacional")

acf(dy,  main = "ACF de la primera diferencia")
acf(dsy, main = "ACF de la diferencia estacional")

# =========================================================
# 5) 4.8.3 MODELIZACIÓN CLÁSICA
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

# -----------------------------
# Modelos benchmark
# -----------------------------
fc_naive  <- naive(y_train, h = h)
fc_snaive <- snaive(y_train, h = h)
fc_drift  <- rwf(y_train, h = h, drift = TRUE)

fit_trend <- tslm(y_train ~ trend)
fc_trend  <- forecast(fit_trend, h = h)

fit_det <- tslm(y_train ~ trend + season)
fc_det  <- forecast(fit_det, h = h)

# -----------------------------
# SARIMA MANUAL PROPUESTO
# SARIMA(1,1,0)(0,0,0)[12]
# -----------------------------
fit_sarima <- arima(
  y_train,
  order = c(1, 1, 0),
  seasonal = list(order = c(0, 0, 0), period = 12),
  include.mean = FALSE,
  method = "ML"
)

fc_sarima <- forecast(fit_sarima, h = h)

sarima_tstats <- fit_sarima$coef / sqrt(diag(fit_sarima$var.coef))

cat("\n================ RESUMEN SARIMA MANUAL ================\n")
print(fit_sarima)
cat("\nT-ratios parámetros SARIMA manual:\n")
print(sarima_tstats)
cat("\nAIC SARIMA manual:", AIC(fit_sarima), "\n")
cat("BIC SARIMA manual:", BIC(fit_sarima), "\n")

# -----------------------------
# AUTOARIMA
# -----------------------------
fit_auto <- auto.arima(
  y_train,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE
)

fc_auto <- forecast(fit_auto, h = h)

cat("\n================ RESUMEN AUTOARIMA ================\n")
print(summary(fit_auto))
cat("AIC AutoARIMA:", AIC(fit_auto), "\n")
cat("BIC AutoARIMA:", BIC(fit_auto), "\n")

# -----------------------------
# ETS y STL-ETS
# -----------------------------
fit_ets <- ets(y_train)
fc_ets  <- forecast(fit_ets, h = h)

fc_stl_ets <- stlf(y_train, h = h, s.window = "periodic", method = "ets")

# -----------------------------
# Diagnóstico residuos SARIMA manual
# -----------------------------
resid_train <- residuals(fit_sarima)

ts.plot(
  resid_train,
  xlab = "Tiempo",
  ylab = "Residuos",
  col = "darkred",
  lwd = 2,
  main = "Residuos del modelo SARIMA manual"
)
abline(h = 0, lty = 2)

acf(resid_train, main = "ACF de los residuos del SARIMA manual")
pacf(resid_train, main = "PACF de los residuos del SARIMA manual")

cat("\n================ LJUNG-BOX SARIMA MANUAL ================\n")
print(Box.test(resid_train, lag = 24, type = "Ljung-Box"))

# -----------------------------
# Ranking modelos clásicos
# -----------------------------
models_classic <- list(
  Naive = fc_naive,
  SeasonalNaive = fc_snaive,
  Drift = fc_drift,
  Trend = fc_trend,
  TrendSeason = fc_det,
  SARIMA_MANUAL = fc_sarima,
  AutoARIMA = fc_auto,
  ETS = fc_ets,
  STL_ETS = fc_stl_ets
)

ranking_classic <- rank_models_test(models_classic, y_test)

cat("\n================ RANKING CLÁSICO EN TEST ================\n")
print(ranking_classic)

ranking_classic$Model <- factor(
  ranking_classic$Model,
  levels = rev(ranking_classic$Model[order(ranking_classic$RMSE, decreasing = TRUE)])
)

p_classic <- ggplot(ranking_classic, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  geom_text(aes(label = round(RMSE, 4)),
            hjust = -0.2,
            size = 4) +
  labs(
    title = "Comparación de modelos clásicos - Tipo de cambio USD/EUR (Test)",
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

sarima_fitted <- y_train - residuals(fit_sarima)

df_plot_sarima <- data.frame(
  time = as.numeric(time(y_train)),
  Real = as.numeric(y_train),
  Ajuste_SARIMA = as.numeric(sarima_fitted)
)

p_sarima <- ggplot(df_plot_sarima, aes(x = time)) +
  geom_line(aes(y = Real, colour = "Serie real"), linewidth = 1) +
  geom_line(aes(y = Ajuste_SARIMA, colour = "Ajuste SARIMA manual"),
            linewidth = 1,
            linetype = "dashed") +
  scale_colour_manual(
    values = c("Serie real" = "black", "Ajuste SARIMA manual" = "red")
  ) +
  labs(
    title = "Ajuste del SARIMA manual sobre tipo de cambio USD/EUR (log)",
    x = "Tiempo",
    y = "log(USD/EUR)",
    colour = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

print(p_sarima)

# =========================================================
# 6) 4.8.4 MODELIZACIÓN MEDIANTE DEEP LEARNING
# =========================================================
grid <- expand.grid(
  type = c("LSTM", "GRU"),
  lookback = c(12, 24, 36),
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
best_dl_fc <- make_forecast_object(y_train, best_fit$fc)

models_usdeur <- models_classic
models_usdeur[[best_dl_name]] <- best_dl_fc

ranking_final <- rank_models_test(models_usdeur, y_test)

cat("\n================ RANKING FINAL EN TEST =================\n")
print(ranking_final)

best_dl_label <- as.character(best_dl_name)

p_final <- ggplot(ranking_final, aes(x = reorder(Model, RMSE), y = RMSE)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = round(RMSE, 4)),
            hjust = -0.2,
            size = 4) +
  labs(
    title = "Comparación final de modelos - Tipo de cambio USD/EUR (Test)",
    x = "Modelo",
    y = "RMSE"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(face = "bold")
  ) +
  ylim(0, max(ranking_final$RMSE) * 1.1)

print(p_final)

ts.plot(
  y,
  col = "black",
  lwd = 2,
  ylab = "log(USD/EUR)",
  main = "Tipo de cambio USD/EUR: Serie real y predicciones finales"
)

lines(fc_sarima$mean, col = "red", lwd = 2, lty = 2)
lines(best_dl_fc$mean, col = "blue", lwd = 2)
lines(fc_stl_ets$mean, col = "darkgreen", lwd = 2)

legend(
  "topleft",
  legend = c("Serie real", "SARIMA manual", best_dl_label, "STL_ETS"),
  col = c("black", "red", "blue", "darkgreen"),
  lty = c(1, 2, 1, 1),
  lwd = 2,
  bty = "n"
)

df_real <- data.frame(
  time = as.numeric(time(y)),
  value = as.numeric(y)
)

df_pred <- data.frame(
  time = as.numeric(time(y_test)),
  value = as.numeric(best_dl_fc$mean)
)

p_dl_usdeur <- ggplot() +
  geom_line(data = df_real,
            aes(x = time, y = value, colour = "Serie real"),
            linewidth = 0.8) +
  geom_line(data = df_pred,
            aes(x = time, y = value, colour = best_dl_label),
            linewidth = 1) +
  scale_colour_manual(
    values = c(best_dl_label = "blue", "Serie real" = "black")
  ) +
  labs(
    title = paste("Predicción del mejor modelo", best_dl_label, "sobre log(USD/EUR)"),
    x = "Tiempo",
    y = "log(USD/EUR)",
    colour = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

print(p_dl_usdeur)

# =========================================================
# 7) 4.8.5 ANÁLISIS DE ESTABILIDAD DEL MODELO DL
# =========================================================
rep_results <- repeat_best_dl(
  y_train = y_train,
  y_test = y_test,
  best_row = best_row,
  n_reps = 10
)

cat("\n================ ESTABILIDAD DEL MEJOR DL ================\n")
print(rep_results)

summary_rep <- data.frame(
  ModelType = as.character(best_row$type),
  RMSE_mean = mean(rep_results$RMSE),
  RMSE_sd   = sd(rep_results$RMSE),
  MAE_mean  = mean(rep_results$MAE),
  MAE_sd    = sd(rep_results$MAE),
  MAPE_mean = mean(rep_results$MAPE),
  MAPE_sd   = sd(rep_results$MAPE)
)

cat("\n================ RESUMEN ESTABILIDAD DL ================\n")
print(summary_rep)

p_stability <- ggplot(rep_results, aes(x = factor(Repetition), y = RMSE)) +
  geom_col() +
  labs(
    title = paste("Estabilidad del modelo", as.character(best_row$type), "- RMSE por repetición"),
    x = "Repetición",
    y = "RMSE"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

print(p_stability)

# =========================================================
# 8) MODELO HÍBRIDO SARIMA + LSTM
# =========================================================
hybrid_fit <- fit_hybrid_residual_lstm(
  fit_sarima = fit_sarima,
  y_train_ts = y_train,
  h = h,
  lookback = 12,
  units = 32,
  dropout = 0,
  batch_size = 8,
  epochs = 200,
  lr = 0.001
)

hybrid_pred <- as.numeric(fc_sarima$mean) + as.numeric(hybrid_fit$resid_fc)
fc_hybrid <- make_forecast_object(
  y_train,
  ts(
    hybrid_pred,
    start = tsp(y_train)[2] + 1 / frequency(y_train),
    frequency = frequency(y_train)
  )
)

comparison_hybrid <- calc_metrics(
  actual = as.numeric(y_test),
  pred   = as.numeric(fc_hybrid$mean)
)
comparison_hybrid$Model <- "SARIMA_MANUAL + LSTM"
comparison_hybrid <- comparison_hybrid[, c("Model", "RMSE", "MAE", "MAPE")]

cat("\n================ RESULTADOS HÍBRIDO ================\n")
print(comparison_hybrid)

ranking_hybrid <- rank_models_test(
  list(
    SARIMA_MANUAL = fc_sarima,
    Hybrid = fc_hybrid
  ),
  y_test
)

cat("\n================ SARIMA MANUAL vs HÍBRIDO ================\n")
print(ranking_hybrid)

p_hybrid <- ggplot(ranking_hybrid, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(RMSE, 4)),
            vjust = -0.3,
            size = 4) +
  labs(
    title = "SARIMA manual vs SARIMA manual + LSTM híbrido - Tipo de cambio USD/EUR",
    x = "Modelo",
    y = "RMSE"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold")) +
  ylim(0, max(ranking_hybrid$RMSE) * 1.1)

print(p_hybrid)

ts.plot(
  y,
  col = "black",
  lwd = 2,
  ylab = "log(USD/EUR)",
  main = "Tipo de cambio USD/EUR: SARIMA manual vs SARIMA+LSTM híbrido"
)

lines(fc_sarima$mean, col = "red", lwd = 2, lty = 2)
lines(fc_hybrid$mean, col = "blue", lwd = 2)

legend(
  "topleft",
  legend = c("Serie real", "SARIMA manual", "SARIMA + LSTM"),
  col = c("black", "red", "blue"),
  lty = c(1, 2, 1),
  lwd = 2,
  bty = "n"
)

plot(
  resid_train,
  main = "Residuos del modelo SARIMA manual",
  ylab = "Residuo",
  col = "darkred"
)
abline(h = 0, lty = 2)

# =========================================================
# 9) CONTRASTE ESTADÍSTICO DE PREDICCIONES
# =========================================================
dm_sarima_bestdl <- dm_compare(
  actual = as.numeric(y_test),
  pred1 = as.numeric(fc_sarima$mean),
  pred2 = as.numeric(best_dl_fc$mean),
  h = 1,
  power = 2
)

dm_sarima_stl <- dm_compare(
  actual = as.numeric(y_test),
  pred1 = as.numeric(fc_sarima$mean),
  pred2 = as.numeric(fc_stl_ets$mean),
  h = 1,
  power = 2
)

dm_sarima_hybrid <- dm_compare(
  actual = as.numeric(y_test),
  pred1 = as.numeric(fc_sarima$mean),
  pred2 = as.numeric(fc_hybrid$mean),
  h = 1,
  power = 2
)

cat("\n================ DIEBOLD-MARIANO =================\n")
cat("\nSARIMA_MANUAL vs BEST_DL\n")
print(dm_sarima_bestdl)

cat("\nSARIMA_MANUAL vs STL_ETS\n")
print(dm_sarima_stl)

cat("\nSARIMA_MANUAL vs HYBRID\n")
print(dm_sarima_hybrid)

# =========================================================
# 10) GRÁFICOS FINALES COMPARATIVOS
# =========================================================
ranking_final_plot <- ranking_final
mejor_clasico <- ranking_classic$Model[1]

ranking_final_plot$Grupo <- "Otros"
ranking_final_plot$Grupo[ranking_final_plot$Model == mejor_clasico] <- "Clásico (Mejor)"
ranking_final_plot$Grupo[grepl("LSTM_BEST|GRU_BEST", ranking_final_plot$Model)] <- "Deep Learning"

ranking_final_plot <- ranking_final_plot[order(ranking_final_plot$RMSE, decreasing = TRUE), ]
ranking_final_plot$Model <- factor(
  ranking_final_plot$Model,
  levels = rev(ranking_final_plot$Model)
)

p_final_colores <- ggplot(ranking_final_plot, aes(x = Model, y = RMSE, fill = Grupo)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = round(RMSE, 4)),
            hjust = -0.15,
            size = 4) +
  scale_fill_manual(
    values = c(
      "Clásico (Mejor)" = "black",
      "Deep Learning" = "blue",
      "Otros" = "grey70"
    )
  ) +
  labs(
    title = "Comparación final de modelos - Tipo de cambio USD/EUR (Test)",
    x = "Modelo",
    y = "RMSE",
    fill = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(face = "bold"),
    legend.position = "top"
  ) +
  ylim(0, max(ranking_final_plot$RMSE) * 1.1)

print(p_final_colores)

ranking_all <- rbind(
  ranking_final,
  data.frame(
    Model = "SARIMA_LSTM_HYBRID",
    RMSE = comparison_hybrid$RMSE,
    MAE  = comparison_hybrid$MAE,
    MAPE = comparison_hybrid$MAPE
  )
)

mejor_clasico <- ranking_classic$Model[1]

ranking_all$Grupo <- "Otros"
ranking_all$Grupo[ranking_all$Model == mejor_clasico] <- "Clásico (Mejor)"
ranking_all$Grupo[grepl("LSTM_BEST|GRU_BEST", ranking_all$Model)] <- "Deep Learning"
ranking_all$Grupo[ranking_all$Model == "SARIMA_LSTM_HYBRID"] <- "Híbrido"

ranking_all <- ranking_all[order(ranking_all$RMSE, decreasing = TRUE), ]
ranking_all$Model <- factor(
  ranking_all$Model,
  levels = rev(ranking_all$Model)
)

p_all <- ggplot(ranking_all, aes(x = Model, y = RMSE, fill = Grupo)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = round(RMSE, 4)),
            hjust = -0.15,
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
    title = "Comparación final completa - Tipo de cambio USD/EUR (Test)",
    x = "Modelo",
    y = "RMSE",
    fill = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(face = "bold"),
    legend.position = "top"
  ) +
  ylim(0, max(ranking_all$RMSE) * 1.1)

print(p_all)