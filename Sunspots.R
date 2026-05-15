# =========================================================
# TFG - CAPÍTULO 4
# SERIE 4.5: SUNSPOT.MONTH
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
# 2.2 Features cíclicas trigonométricas
# IMPORTANTE:
# Sunspot.month es mensual, pero no presenta una
# estacionalidad anual regular como CO2.
# Aproximamos el ciclo solar a 11 años = 132 meses.
# -----------------------------
make_cycle_features <- function(y_ts, period = 132){
  t <- seq_along(y_ts)
  sin_c <- sin(2 * pi * t / period)
  cos_c <- cos(2 * pi * t / period)
  cbind(sin_c = sin_c, cos_c = cos_c)
}

# -----------------------------
# 2.3 Construcción de ventanas multivariantes
# feature 1 = serie escalada
# feature 2 = sin ciclo
# feature 3 = cos ciclo
# -----------------------------
make_windows_multifeature <- function(y_scaled, cycle_mat, lookback){
  n <- length(y_scaled)
  if(n <= lookback) stop("No hay suficientes datos para crear ventanas.")
  
  X <- array(NA_real_, dim = c(n - lookback, lookback, 3))
  y <- numeric(n - lookback)
  
  for(i in seq_len(n - lookback)){
    idx <- i:(i + lookback - 1)
    X[i, , 1] <- y_scaled[idx]
    X[i, , 2] <- cycle_mat[idx, 1]
    X[i, , 3] <- cycle_mat[idx, 2]
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
                                            cycle_hist,
                                            cycle_future,
                                            h,
                                            lookback){
  preds <- numeric(h)
  y_ext <- as.numeric(y_hist_scaled)
  cycle_ext <- cycle_hist
  
  for(i in 1:h){
    idx <- (length(y_ext) - lookback + 1):length(y_ext)
    
    x_new <- array(NA_real_, dim = c(1, lookback, 3))
    x_new[1, , 1] <- y_ext[idx]
    x_new[1, , 2] <- cycle_ext[idx, 1]
    x_new[1, , 3] <- cycle_ext[idx, 2]
    
    pred_i <- as.numeric(predict(model, x_new, verbose = 0))
    preds[i] <- pred_i
    
    y_ext <- c(y_ext, pred_i)
    cycle_ext <- rbind(cycle_ext, cycle_future[i, , drop = FALSE])
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
                                        verbose_fit = 0,
                                        cycle_period = 132){
  
  type <- match.arg(type)
  
  sc <- minmax_scale_train(as.numeric(y_train_ts))
  y_scaled <- sc$x_s
  
  cycle_train <- make_cycle_features(y_train_ts, period = cycle_period)
  windows <- make_windows_multifeature(y_scaled, cycle_train, lookback)
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
  
  t_hist <- seq_along(y_train_ts)
  t_future <- (length(t_hist) + 1):(length(t_hist) + h)
  
  cycle_future <- cbind(
    sin_c = sin(2 * pi * t_future / cycle_period),
    cos_c = cos(2 * pi * t_future / cycle_period)
  )
  
  fc_scaled <- recursive_forecast_multifeature(
    model = model,
    y_hist_scaled = y_scaled,
    cycle_hist = cycle_train,
    cycle_future = cycle_future,
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
repeat_best_dl <- function(y_train, y_test, best_row, n_reps = 10, cycle_period = 132){
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
      verbose_fit = 0,
      cycle_period = cycle_period
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
# 2.12 Híbrido ARIMA + LSTM sobre residuos
# -----------------------------
fit_hybrid_residual_lstm <- function(fit_arima, y_train_ts, h,
                                     lookback = 60,
                                     units = 32,
                                     dropout = 0,
                                     batch_size = 8,
                                     epochs = 200,
                                     lr = 0.001,
                                     cycle_period = 132){
  
  resid_train <- residuals(fit_arima)
  resid_train <- as.numeric(resid_train)
  
  resid_train_ts <- ts(
    resid_train,
    start = start(y_train_ts),
    frequency = frequency(y_train_ts)
  )
  
  sc <- minmax_scale_train(resid_train)
  y_scaled <- sc$x_s
  
  cycle_train <- make_cycle_features(resid_train_ts, period = cycle_period)
  windows <- make_windows_multifeature(y_scaled, cycle_train, lookback)
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
  
  t_hist <- seq_along(resid_train_ts)
  t_future <- (length(t_hist) + 1):(length(t_hist) + h)
  
  cycle_future <- cbind(
    sin_c = sin(2 * pi * t_future / cycle_period),
    cos_c = cos(2 * pi * t_future / cycle_period)
  )
  
  resid_pred_scaled <- recursive_forecast_multifeature(
    model = model,
    y_hist_scaled = y_scaled,
    cycle_hist = cycle_train,
    cycle_future = cycle_future,
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
# 3) 4.5.1 DESCRIPCIÓN DE LA SERIE
# =========================================================
y0 <- sunspot.month

cat("\n=====================================================\n")
cat("4.5.1 DESCRIPCIÓN DE LA SERIE: SUNSPOT.MONTH\n")
cat("=====================================================\n")
cat("Frecuencia:", frequency(y0), "\n")
cat("Inicio:", paste(start(y0), collapse = "-"), "\n")
cat("Fin:", paste(end(y0), collapse = "-"), "\n")
cat("Número de observaciones:", length(y0), "\n")

# =========================================================
# 4) 4.5.2 ANÁLISIS EXPLORATORIO Y TRANSFORMACIONES
# =========================================================
plot(
  y0,
  main = "Sunspot.month - Serie original",
  ylab = "Número mensual de manchas solares",
  col = "black",
  lwd = 2
)

# Transformación robusta a ceros
y <- log1p(y0)

plot(
  y,
  main = "Sunspot.month - Serie en log(1+x)",
  ylab = "log(1 + sunspots)",
  col = "blue",
  lwd = 2
)

# Se puede visualizar el patrón por meses aunque no sea estacionalidad anual "pura"
ggseasonplot(y0, year.labels = FALSE) +
  ylab("Sunspots") +
  ggtitle("Seasonal plot de Sunspot.month")

acf(y, main = "ACF de log(1 + Sunspot.month)")
pacf(y, main = "PACF de log(1 + Sunspot.month)")

dy <- diff(y, differences = 1)

plot(
  dy,
  main = "Primera diferencia de log(1 + Sunspot.month)",
  ylab = "Diferencia",
  col = "darkgreen",
  lwd = 2
)

acf(dy, main = "ACF de la primera diferencia")
pacf(dy, main = "PACF de la primera diferencia")



ggsubseriesplot(log1p(sunspot.month)) +
  ylab("log(1 + Sunspots)") +
  xlab("Mes") +
  ggtitle("Seasonal subseries plot de Sunspot.month") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.title = element_text(face = "bold")
  )
# =========================================================
# 5) 4.5.3 MODELIZACIÓN CLÁSICA
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

fc_naive  <- naive(y_train, h = h)
fc_snaive <- snaive(y_train, h = h)
fc_drift  <- rwf(y_train, h = h, drift = TRUE)

fit_trend <- tslm(y_train ~ trend)
fc_trend  <- forecast(fit_trend, h = h)

fit_trend_season <- tslm(y_train ~ trend + season)
fc_trend_season  <- forecast(fit_trend_season, h = h)

# Modelo manual inicial
fit_arima <- Arima(
  y_train,
  order = c(3, 1, 3)
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

fit_stl <- stlm(y_train, modelfunction = ets)
fc_stl_ets <- forecast(fit_stl, h = h)

cat("\n================ RESUMEN ARIMA MANUAL ================\n")
print(summary(fit_arima))
cat("AIC ARIMA manual:", AIC(fit_arima), "\n")
cat("BIC ARIMA manual:", BIC(fit_arima), "\n")

cat("\n================ RESUMEN AUTOARIMA ================\n")
print(summary(fit_auto))
cat("AIC AutoARIMA:", AIC(fit_auto), "\n")
cat("BIC AutoARIMA:", BIC(fit_auto), "\n")

resid_train <- residuals(fit_arima)

ts.plot(
  resid_train,
  xlab = "Tiempo",
  ylab = "Residuos",
  col = "darkred",
  lwd = 2,
  main = "Residuos del modelo ARIMA manual"
)
abline(h = 0, lty = 2)

acf(resid_train, main = "ACF de los residuos del ARIMA manual")
pacf(resid_train, main = "PACF de los residuos del ARIMA manual")

cat("\n================ LJUNG-BOX ARIMA MANUAL ================\n")
print(Box.test(resid_train, lag = 24, type = "Ljung-Box"))

models_classic <- list(
  Naive = fc_naive,
  SNaive = fc_snaive,
  Drift = fc_drift,
  Trend = fc_trend,
  Trend_Season = fc_trend_season,
  ARIMA_MANUAL = fc_arima,
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
    title = "Comparación de modelos clásicos - Sunspot.month (Test)",
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
    title = "Ajuste del modelo ARIMA manual sobre Sunspot.month",
    x = "Tiempo",
    y = "log(1 + Sunspots)",
    colour = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

print(p_arima)


# =========================
# Zoom final del ajuste ARIMA manual
# =========================
n_zoom <- 180   # últimos 180 meses; puedes probar 120 o 240

df_plot_arima_zoom <- tail(df_plot_arima, n_zoom)

p_arima_zoom <- ggplot(df_plot_arima_zoom, aes(x = time)) +
  geom_line(aes(y = Real, colour = "Serie real"), linewidth = 0.7) +
  geom_line(
    aes(y = Ajuste_ARIMA, colour = "Ajuste ARIMA"),
    linewidth = 0.9,
    linetype = "dashed"
  ) +
  scale_colour_manual(
    values = c(
      "Serie real" = "black",
      "Ajuste ARIMA" = "red"
    )
  ) +
  labs(
    title = "Ajuste del modelo ARIMA manual sobre Sunspot.month (tramo final)",
    x = "Tiempo",
    y = "log(1 + Sunspots)",
    colour = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

print(p_arima_zoom)

# =========================================================
# 6) 4.5.4 MODELIZACIÓN MEDIANTE DEEP LEARNING
# =========================================================
grid <- expand.grid(
  type = c("LSTM", "GRU"),
  lookback = c(24, 60, 132),
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
    verbose_fit = 0,
    cycle_period = 132
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

models_sunspots <- models_classic
models_sunspots[[best_dl_name]] <- best_dl_fc

ranking_final <- rank_models_test(models_sunspots, y_test)

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
    title = "Comparación final de modelos - Sunspot.month (Test)",
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
  ylab = "log(1 + Sunspots)",
  main = "Sunspot.month: Serie real y predicciones finales"
)

lines(fc_arima$mean, col = "red", lwd = 2, lty = 2)
lines(best_dl_fc$mean, col = "blue", lwd = 2)
lines(fc_stl_ets$mean, col = "darkgreen", lwd = 2)

legend(
  "topleft",
  legend = c("Serie real", "ARIMA_MANUAL", best_dl_label, "STL_ETS"),
  col = c("black", "red", "blue", "darkgreen"),
  lty = c(1, 2, 1, 1),
  lwd = 2,
  bty = "n"
)




# =========================
# Zoom del tramo final
# =========================
n_zoom <- 180  # últimos 180 meses, por ejemplo

y_zoom <- window(
  y,
  start = time(y)[length(y) - n_zoom + 1]
)

df_real_zoom <- data.frame(
  time = as.numeric(time(y_zoom)),
  value = as.numeric(y_zoom),
  Serie = "Serie real"
)

df_fc_zoom <- data.frame(
  time = as.numeric(time(best_dl_fc$mean)),
  value = as.numeric(best_dl_fc$mean),
  Serie = as.character(best_row$type)
)

p_best_dl_zoom <- ggplot() +
  geom_line(
    data = df_real_zoom,
    aes(x = time, y = value, colour = Serie),
    linewidth = 0.7
  ) +
  geom_line(
    data = df_fc_zoom,
    aes(x = time, y = value, colour = Serie),
    linewidth = 1.2
  ) +
  scale_colour_manual(
    values = c(
      "Serie real" = "black",
      "LSTM" = "blue",
      "GRU" = "blue"
    )
  ) +
  labs(
    title = paste("Predicción del mejor modelo", as.character(best_row$type), "sobre log(1 + Sunspot.month)"),
    x = "Tiempo",
    y = "log(1 + Sunspots)",
    colour = ""
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

print(p_best_dl_zoom)
# =========================================================
# 7) ANÁLISIS DE ESTABILIDAD DEL MODELO DL
# =========================================================
rep_results <- repeat_best_dl(
  y_train = y_train,
  y_test = y_test,
  best_row = best_row,
  n_reps = 10,
  cycle_period = 132
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
# 8) MODELO HÍBRIDO ARIMA + LSTM
# =========================================================
hybrid_fit <- fit_hybrid_residual_lstm(
  fit_arima = fit_arima,
  y_train_ts = y_train,
  h = h,
  lookback = 60,
  units = 32,
  dropout = 0,
  batch_size = 8,
  epochs = 200,
  lr = 0.001,
  cycle_period = 132
)

hybrid_pred <- as.numeric(fc_arima$mean) + as.numeric(hybrid_fit$resid_fc)
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
comparison_hybrid$Model <- "ARIMA + LSTM"
comparison_hybrid <- comparison_hybrid[, c("Model", "RMSE", "MAE", "MAPE")]

cat("\n================ RESULTADOS HÍBRIDO ================\n")
print(comparison_hybrid)

ranking_hybrid <- rank_models_test(
  list(
    ARIMA_MANUAL = fc_arima,
    Hybrid = fc_hybrid
  ),
  y_test
)

cat("\n================ ARIMA vs HÍBRIDO ================\n")
print(ranking_hybrid)

p_hybrid <- ggplot(ranking_hybrid, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(RMSE, 4)),
            vjust = -0.3,
            size = 4) +
  labs(
    title = "ARIMA manual vs ARIMA + LSTM híbrido - Sunspot.month",
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
  ylab = "log(1 + Sunspots)",
  main = "Sunspot.month: ARIMA vs ARIMA+LSTM híbrido"
)

lines(fc_arima$mean, col = "red", lwd = 2, lty = 2)
lines(fc_hybrid$mean, col = "blue", lwd = 2)

legend(
  "topleft",
  legend = c("Serie real", "ARIMA manual", "ARIMA + LSTM"),
  col = c("black", "red", "blue"),
  lty = c(1, 2, 1),
  lwd = 2,
  bty = "n"
)

plot(
  resid_train,
  main = "Residuos del modelo ARIMA manual",
  ylab = "Residuo",
  col = "darkred"
)
abline(h = 0, lty = 2)





# =========================
# Zoom final aruma vs arima hibrido
# =========================

n_zoom <- 180   # prueba también 120

y_zoom <- window(
  y,
  start = time(y)[length(y) - n_zoom + 1]
)

plot(
  y_zoom,
  type = "l",
  col = "black",
  lwd = 1.2,
  xlab = "Time",
  ylab = "log(1 + Sunspots)",
  main = "log(Sunspot.month): ARIMA vs ARIMA+LSTM híbrido"
)

lines(
  fc_arima$mean,
  col = "red",
  lwd = 1.5,
  lty = 2
)

lines(
  fc_hybrid$mean,
  col = "blue",
  lwd = 1.5,
  lty = 1
)

legend(
  "topleft",
  legend = c("Serie real", "ARIMA", "ARIMA + LSTM"),
  col = c("black", "red", "blue"),
  lty = c(1, 2, 1),
  lwd = c(1.2, 1.5, 1.5),
  bty = "n"
)
# =========================================================
# 9) CONTRASTE ESTADÍSTICO DE PREDICCIONES
# =========================================================
dm_arima_bestdl <- dm_compare(
  actual = as.numeric(y_test),
  pred1 = as.numeric(fc_arima$mean),
  pred2 = as.numeric(best_dl_fc$mean),
  h = 1,
  power = 2
)

dm_arima_stl <- dm_compare(
  actual = as.numeric(y_test),
  pred1 = as.numeric(fc_arima$mean),
  pred2 = as.numeric(fc_stl_ets$mean),
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
cat("\nARIMA_MANUAL vs BEST_DL\n")
print(dm_arima_bestdl)

cat("\nARIMA_MANUAL vs STL_ETS\n")
print(dm_arima_stl)

cat("\nARIMA_MANUAL vs HYBRID\n")
print(dm_arima_hybrid)

# =========================================================
# 10) GRÁFICOS FINALES DE COMPARACIÓN
# =========================================================
ranking_final_plot <- ranking_final

ranking_final_plot$Grupo <- "Otros"
ranking_final_plot$Grupo[ranking_final_plot$Model == "ARIMA_MANUAL"] <- "Clásico (Mejor)"
ranking_final_plot$Grupo[ranking_final_plot$Model %in% c("LSTM_BEST", "GRU_BEST")] <- "Deep Learning"

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
    title = "Comparación final de modelos - Sunspot.month (Test)",
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
    Model = "ARIMA_LSTM_HYBRID",
    RMSE = comparison_hybrid$RMSE,
    MAE  = comparison_hybrid$MAE,
    MAPE = comparison_hybrid$MAPE
  )
)

ranking_all$Grupo <- "Otros"
ranking_all$Grupo[ranking_all$Model == "ARIMA_MANUAL"] <- "Clásico (Mejor)"
ranking_all$Grupo[ranking_all$Model %in% c("LSTM_BEST", "GRU_BEST")] <- "Deep Learning"
ranking_all$Grupo[ranking_all$Model == "ARIMA_LSTM_HYBRID"] <- "Híbrido"

ranking_all <- ranking_all[order(ranking_all$RMSE, decreasing = TRUE), ]
ranking_all$Model <- factor(ranking_all$Model, levels = rev(ranking_all$Model))

p_final_hybrid <- ggplot(ranking_all, aes(x = Model, y = RMSE, fill = Grupo)) +
  geom_col() +
  coord_flip() +
  geom_text(aes(label = round(RMSE, 4)),
            hjust = -0.15,
            size = 4) +
  scale_fill_manual(
    values = c(
      "Clásico (Mejor)" = "black",
      "Deep Learning"   = "blue",
      "Híbrido"         = "green4",
      "Otros"           = "grey70"
    )
  ) +
  labs(
    title = "Comparación final de modelos (incluyendo híbrido) - Sunspot.month",
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

print(p_final_hybrid)

# =========================================================
# 11) EXPORTACIÓN DE RESULTADOS
# =========================================================
write.csv(ranking_classic, "sunspot_month_ranking_classic.csv", row.names = FALSE)
write.csv(results_df, "sunspot_month_dl_grid_results.csv", row.names = FALSE)
write.csv(ranking_final, "sunspot_month_final_ranking.csv", row.names = FALSE)
write.csv(rep_results, "sunspot_month_dl_repeticiones.csv", row.names = FALSE)
write.csv(summary_rep, "sunspot_month_dl_repeticiones_resumen.csv", row.names = FALSE)
write.csv(comparison_hybrid, "sunspot_month_arima_vs_hybrid.csv", row.names = FALSE)
write.csv(ranking_hybrid, "sunspot_month_hybrid_ranking.csv", row.names = FALSE)

# =========================================================
# FIN DEL SCRIPT
# =========================================================



# =========================
# Zoom final estilo Nile
# =========================

n_zoom <- 180   # prueba también 120

y_zoom <- window(
  y,
  start = time(y)[length(y) - n_zoom + 1]
)

plot(
  y_zoom,
  type = "l",
  col = "black",
  lwd = 1.2,
  xlab = "Time",
  ylab = "log(1 + Sunspots)",
  main = "log(Sunspot.month): ARIMA vs ARIMA+LSTM híbrido"
)

lines(
  fc_arima$mean,
  col = "red",
  lwd = 1.5,
  lty = 2
)

lines(
  fc_hybrid$mean,
  col = "blue",
  lwd = 1.5,
  lty = 1
)

legend(
  "topleft",
  legend = c("Serie real", "ARIMA", "ARIMA + LSTM"),
  col = c("black", "red", "blue"),
  lty = c(1, 2, 1),
  lwd = c(1.2, 1.5, 1.5),
  bty = "n"
)