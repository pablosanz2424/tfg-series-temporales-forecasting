# =========================================================
# TFG - CAPÍTULO 4
# SERIE 4.9: DEMANDA ELÉCTRICA
# =========================================================

# =========================================================
# 0) PAQUETES
# =========================================================

req <- c("forecast", "keras3", "tensorflow", "Metrics", "ggplot2", "dplyr", "lubridate")
new <- setdiff(req, rownames(installed.packages()))
if(length(new)) install.packages(new, dependencies = TRUE)

library(forecast)
library(keras3)
library(tensorflow)
library(Metrics)
library(ggplot2)
library(dplyr)


# =========================================================
# 1) SEMILLA REPRODUCIBLE
# =========================================================

set.seed(123)
tensorflow::set_random_seed(123)

# =========================================================
# 2) FUNCIONES AUXILIARES GENERALES
# =========================================================

minmax_scale_train <- function(x){
  xmin <- min(x, na.rm = TRUE)
  xmax <- max(x, na.rm = TRUE)
  xs <- (x - xmin) / (xmax - xmin)
  list(x_s = as.numeric(xs), min = xmin, max = xmax)
}

minmax_invert <- function(x_s, xmin, xmax){
  as.numeric(x_s * (xmax - xmin) + xmin)
}

make_seasonal_features <- function(y_ts){
  s <- frequency(y_ts)
  t <- seq_along(y_ts)
  sin_s <- sin(2 * pi * t / s)
  cos_s <- cos(2 * pi * t / s)
  cbind(sin_s = sin_s, cos_s = cos_s)
}

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

calc_metrics <- function(actual, pred){
  data.frame(
    RMSE = Metrics::rmse(actual, pred),
    MAE  = Metrics::mae(actual, pred),
    MAPE = mean(abs((actual - pred) / actual)) * 100
  )
}

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

build_rnn_model <- function(input_shape,
                            type = c("LSTM", "GRU"),
                            units = 32,
                            dropout = 0,
                            lr = 0.001){
  
  type <- match.arg(type)
  
  inputs <- layer_input(shape = input_shape)
  
  if(type == "LSTM"){
    x <- inputs |>
      layer_lstm(
        units = units,
        dropout = dropout,
        recurrent_dropout = 0
      )
  } else {
    x <- inputs |>
      layer_gru(
        units = units,
        dropout = dropout,
        recurrent_dropout = 0
      )
  }
  
  outputs <- x |>
    layer_dense(units = 1)
  
  model <- keras_model(
    inputs = inputs,
    outputs = outputs
  )
  
  model |>
    compile(
      loss = "mse",
      optimizer = optimizer_adam(learning_rate = lr),
      metrics = list("mae")
    )
  
  return(model)
}

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

repeat_best_dl <- function(y_train, y_test, best_row, n_reps = 10){
  out <- list()
  
  best_type <- as.character(best_row$type)
  best_lookback <- as.numeric(best_row$lookback)
  best_units <- as.numeric(best_row$units)
  best_dropout <- as.numeric(best_row$dropout)
  best_batch <- as.numeric(best_row$batch_size)
  
  for(i in 1:n_reps){
    set.seed(100 + i)
    tensorflow::set_random_seed(100 + i)
    
    fit_i <- fit_rnn_univariate_improved(
      y_train_ts = y_train,
      h = length(y_test),
      lookback = best_lookback,
      type = best_type,
      units = best_units,
      dropout = best_dropout,
      batch_size = best_batch,
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
      ModelType = best_type,
      RMSE = met_i$RMSE,
      MAE  = met_i$MAE,
      MAPE = met_i$MAPE
    )
  }
  
  rep_df <- do.call(rbind, out)
  rownames(rep_df) <- NULL
  rep_df
}

dm_compare <- function(actual, pred1, pred2, h = 1, power = 2){
  e1 <- actual - pred1
  e2 <- actual - pred2
  forecast::dm.test(e1, e2, h = h, power = power)
}

make_forecast_object <- function(y_train, y_pred){
  fc <- forecast::forecast(y_train, h = length(y_pred))
  fc$mean <- y_pred
  fc
}


fit_hybrid_residual_lstm <- function(fit_sarima,
                                     y_train_ts,
                                     h,
                                     lookback = 14,
                                     units = 32,
                                     dropout = 0,
                                     batch_size = 8,
                                     epochs = 200,
                                     lr = 0.001){
  
  resid_train <- as.numeric(residuals(fit_sarima))
  resid_train[is.na(resid_train)] <- 0
  
  resid_ts <- ts(
    resid_train,
    start = start(y_train_ts),
    frequency = frequency(y_train_ts)
  )
  
  fit_resid <- fit_rnn_univariate_improved(
    y_train_ts = resid_ts,
    h = h,
    lookback = lookback,
    type = "LSTM",
    units = units,
    dropout = dropout,
    batch_size = batch_size,
    epochs = epochs,
    lr = lr,
    val_prop = 0.2,
    patience = 15,
    verbose_fit = 0
  )
  
  list(
    resid_fc = fit_resid$fc,
    model = fit_resid$model,
    history = fit_resid$history
  )
}

# =========================================================
# 3) 4.9.1 DESCRIPCIÓN DE LA SERIE
# =========================================================

df <- DEMANDA_ELECTRICA

names(df) <- tolower(names(df))

df <- df %>%
  mutate(
    fecha = as.character(fecha),
    fecha = tolower(fecha),
    fecha = gsub("ene", "01", fecha),
    fecha = gsub("feb", "02", fecha),
    fecha = gsub("mar", "03", fecha),
    fecha = gsub("abr", "04", fecha),
    fecha = gsub("may", "05", fecha),
    fecha = gsub("jun", "06", fecha),
    fecha = gsub("jul", "07", fecha),
    fecha = gsub("ago", "08", fecha),
    fecha = gsub("sep", "09", fecha),
    fecha = gsub("oct", "10", fecha),
    fecha = gsub("nov", "11", fecha),
    fecha = gsub("dic", "12", fecha),
    fecha = as.Date(fecha, format = "%d/%m/%y"),
    demanda = as.numeric(demanda)
  ) %>%
  arrange(fecha) %>%
  filter(!is.na(fecha), !is.na(demanda))

# Objeto ts para modelización
# frequency = 7 porque se considera estacionalidad semanal en datos diarios
y0 <- ts(df$demanda, start = c(2024, 1), frequency = 7)

cat("\n=====================================================\n")
cat("4.X.1 DESCRIPCIÓN DE LA SERIE: DEMANDA ELÉCTRICA\n")
cat("=====================================================\n")
cat("Frecuencia:", frequency(y0), "(datos diarios con estacionalidad semanal)\n")
cat("Inicio:", as.character(min(df$fecha)), "\n")
cat("Fin:", as.character(max(df$fecha)), "\n")
cat("Número de observaciones:", length(y0), "\n")

# =========================================================
# 4) 4.9.2 ANÁLISIS EXPLORATORIO Y TRANSFORMACIONES
# =========================================================

# =========================================================
# 4.1 GRÁFICO DE LA SERIE ORIGINAL CON FECHAS REALES
# =========================================================

plot(
  df$fecha,
  df$demanda,
  type = "l",
  main = "Demanda eléctrica - Serie original",
  xlab = "Fecha",
  ylab = "Demanda eléctrica",
  col = "black",
  lwd = 2
)

if(any(y0 <= 0, na.rm = TRUE)){
  stop("La serie contiene valores no positivos. No procede aplicar logaritmos directamente.")
}

y <- log(y0)

df$log_demanda <- as.numeric(y)

# =========================================================
# 4.2 GRÁFICO DE LA SERIE EN LOGARITMOS CON FECHAS REALES
# =========================================================

plot(
  df$fecha,
  df$log_demanda,
  type = "l",
  main = "Demanda eléctrica - Serie en logaritmos",
  xlab = "Fecha",
  ylab = "log(demanda eléctrica)",
  col = "blue",
  lwd = 2
)

# =========================================================
# 4.3 DÍA DE LA SEMANA
# Sin necesidad de usar lubridate
# =========================================================

dia_num <- as.POSIXlt(df$fecha)$wday
dia_num <- ifelse(dia_num == 0, 7, dia_num)

df$dia_semana <- factor(
  dia_num,
  levels = 1:7,
  labels = c("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
)

ggplot(df, aes(x = dia_semana, y = demanda)) +
  geom_boxplot() +
  labs(
    title = "Demanda eléctrica por día de la semana",
    x = "Día de la semana",
    y = "Demanda eléctrica"
  ) +
  theme_minimal(base_size = 13)

ggplot(df, aes(x = dia_semana, y = log_demanda)) +
  geom_boxplot() +
  labs(
    title = "log(demanda eléctrica) por día de la semana",
    x = "Día de la semana",
    y = "log(demanda eléctrica)"
  ) +
  theme_minimal(base_size = 13)

# =========================================================
# 4.4 ACF Y PACF
# Estos gráficos no usan fechas, sino retardos
# =========================================================

acf(y, main = "ACF de log(demanda eléctrica)", lag.max = 60)
pacf(y, main = "PACF de log(demanda eléctrica)", lag.max = 60)

# =========================================================
# 4.5 DIFERENCIAS CON FECHAS REALES
# =========================================================

dy  <- diff(y, differences = 1)
dsy <- diff(y, lag = frequency(y), differences = 1)

fechas_dy  <- df$fecha[-1]
fechas_dsy <- df$fecha[-seq_len(frequency(y))]

plot(
  fechas_dy,
  as.numeric(dy),
  type = "l",
  main = "Primera diferencia de log(demanda eléctrica)",
  xlab = "Fecha",
  ylab = "Diferencia",
  col = "black",
  lwd = 2
)
abline(h = 0, lty = 2)

plot(
  fechas_dsy,
  as.numeric(dsy),
  type = "l",
  main = "Diferencia semanal de log(demanda eléctrica)",
  xlab = "Fecha",
  ylab = "Diferencia semanal",
  col = "black",
  lwd = 2
)
abline(h = 0, lty = 2)

acf(dy,  main = "ACF de la primera diferencia", lag.max = 60)
acf(dsy, main = "ACF de la diferencia semanal", lag.max = 60)

# =========================================================
# 5) 4.9.3 MODELIZACIÓN CLÁSICA
# =========================================================

h <- 7
n <- length(y)

y_train <- window(y, end = time(y)[n - h])
y_test  <- window(y, start = time(y)[n - h + 1])

dates_train <- df$fecha[1:(n - h)]
dates_test  <- df$fecha[(n - h + 1):n]

cat("\n=====================================================\n")
cat("DIVISIÓN TRAIN / TEST\n")
cat("=====================================================\n")
cat("Observaciones train:", length(y_train), "\n")
cat("Observaciones test:", length(y_test), "\n")
cat("Horizonte de predicción h:", h, "días\n")
cat("Inicio test:", as.character(min(dates_test)), "\n")
cat("Fin test:", as.character(max(dates_test)), "\n")

# =========================================================
# 5.1 MODELOS BENCHMARK Y CLÁSICOS
# =========================================================

fc_naive  <- naive(y_train, h = h)
fc_snaive <- snaive(y_train, h = h)
fc_drift  <- rwf(y_train, h = h, drift = TRUE)

fit_trend <- tslm(y_train ~ trend)
fc_trend  <- forecast(fit_trend, h = h)

fit_det <- tslm(y_train ~ trend + season)
fc_det  <- forecast(fit_det, h = h)

# En esta serie se usa el modelo seleccionado automáticamente,
# dado que la inspección gráfica no permite fijar claramente una estructura SARIMA manual.
fit_sarima <- auto.arima(
  y_train,
  seasonal = TRUE,
  stepwise = FALSE,
  approximation = FALSE
)

fc_sarima <- forecast(fit_sarima, h = h)

# Se mantiene como alias para conservar la estructura del resto del trabajo
fit_auto <- fit_sarima
fc_auto  <- fc_sarima

fit_ets <- ets(y_train)
fc_ets  <- forecast(fit_ets, h = h)

fc_stl_ets <- stlf(y_train, h = h, s.window = "periodic", method = "ets")

cat("\n================ RESUMEN SARIMA AUTOMÁTICO ================\n")
print(summary(fit_sarima))
cat("AIC SARIMA:", AIC(fit_sarima), "\n")
cat("BIC SARIMA:", BIC(fit_sarima), "\n")

cat("\n================ RESUMEN AUTOARIMA ================\n")
print(summary(fit_auto))
cat("AIC AutoARIMA:", AIC(fit_auto), "\n")
cat("BIC AutoARIMA:", BIC(fit_auto), "\n")

# =========================================================
# 5.2 RESIDUOS DEL SARIMA CON FECHAS REALES
# =========================================================

resid_train <- residuals(fit_sarima)

df_resid <- data.frame(
  fecha = dates_train,
  residuo = as.numeric(resid_train)
)

plot(
  df_resid$fecha,
  df_resid$residuo,
  type = "l",
  xlab = "Fecha",
  ylab = "Residuos",
  col = "darkred",
  lwd = 2,
  main = "Residuos del modelo SARIMA - Demanda eléctrica"
)
abline(h = 0, lty = 2)

acf(resid_train, main = "ACF de los residuos del SARIMA", lag.max = 60)
pacf(resid_train, main = "PACF de los residuos del SARIMA", lag.max = 60)

cat("\n================ LJUNG-BOX SARIMA ================\n")
print(Box.test(resid_train, lag = 28, type = "Ljung-Box"))

# =========================================================
# 5.3 COMPARACIÓN DE MODELOS CLÁSICOS
# =========================================================

models_classic <- list(
  Naive = fc_naive,
  SeasonalNaive = fc_snaive,
  Drift = fc_drift,
  Trend = fc_trend,
  TrendSeason = fc_det,
  SARIMA = fc_sarima,
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
  geom_text(aes(label = round(RMSE, 4)), hjust = -0.2, size = 4) +
  labs(
    title = "Comparación de modelos clásicos - Demanda eléctrica (Test)",
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

# =========================================================
# 5.4 AJUSTE DEL MODELO SARIMA CON FECHAS REALES
# =========================================================

ajuste_sarima <- as.numeric(fitted(fit_sarima))

len_fit <- min(
  length(dates_train),
  length(as.numeric(y_train)),
  length(ajuste_sarima)
)

df_plot_sarima <- data.frame(
  fecha = dates_train[seq_len(len_fit)],
  Real = as.numeric(y_train)[seq_len(len_fit)],
  Ajuste_SARIMA = ajuste_sarima[seq_len(len_fit)]
)

p_sarima <- ggplot(df_plot_sarima, aes(x = fecha)) +
  geom_line(
    aes(y = Real, colour = "Serie real"),
    linewidth = 1
  ) +
  geom_line(
    aes(y = Ajuste_SARIMA, colour = "Ajuste SARIMA"),
    linewidth = 1,
    linetype = "dashed"
  ) +
  scale_colour_manual(
    values = c("Serie real" = "black", "Ajuste SARIMA" = "red")
  ) +
  labs(
    title = "Ajuste del modelo SARIMA sobre demanda eléctrica (log)",
    x = "Fecha",
    y = "log(demanda eléctrica)",
    colour = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

print(p_sarima)
# =========================================================
# 6) 4.9.4 MODELIZACIÓN MEDIANTE DEEP LEARNING
# =========================================================

grid <- expand.grid(
  type = c("LSTM", "GRU"),
  lookback = c(7, 14, 28),
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

# =========================================================
# ETIQUETAS DINÁMICAS DEL MEJOR MODELO DL
# =========================================================

best_dl_label <- as.character(best_row$type)
best_dl_name  <- paste0(best_dl_label, "_BEST")

best_idx <- which(
  grid$type == best_row$type &
    grid$lookback == best_row$lookback &
    grid$units == best_row$units &
    grid$dropout == best_row$dropout &
    grid$batch_size == best_row$batch_size
)[1]

best_fit <- fits[[best_idx]]
best_dl_fc <- make_forecast_object(y_train, best_fit$fc)

models_demanda <- models_classic
models_demanda[[best_dl_name]] <- best_dl_fc

ranking_final <- rank_models_test(models_demanda, y_test)

# =========================================================
# GRUPO DE COLOR PARA EL GRÁFICO
# =========================================================

ranking_final$Grupo <- "Otros"
ranking_final$Grupo[ranking_final$Model == best_dl_name] <- "Mejor DL"
ranking_final$Grupo[ranking_final$Model == "STL_ETS"]    <- "STL_ETS"

# =========================================================
# ORDEN: PEORES ARRIBA, MEJORES ABAJO
# =========================================================

ranking_final <- ranking_final[order(ranking_final$RMSE, decreasing = FALSE), ]

ranking_final$Model <- factor(
  ranking_final$Model,
  levels = ranking_final$Model
)

# =========================================================
# GRÁFICO FINAL COMPARATIVO
# =========================================================

p_final <- ggplot(ranking_final, aes(x = Model, y = RMSE, fill = Grupo)) +
  geom_col() +
  coord_flip() +
  geom_text(
    aes(label = round(RMSE, 4)),
    hjust = -0.2,
    size = 4
  ) +
  scale_fill_manual(
    values = c(
      "Mejor DL" = "blue",
      "STL_ETS"  = "black",
      "Otros"    = "grey40"
    )
  ) +
  labs(
    title = "Comparación final de modelos - Demanda eléctrica (Test)",
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
  ylim(0, max(ranking_final$RMSE) * 1.1)

print(p_final)

# =========================================================
# GRÁFICO BASE: SERIE REAL Y PREDICCIONES FINALES
# =========================================================

ts.plot(
  y,
  col = "black",
  lwd = 2,
  ylab = "log(demanda eléctrica)",
  main = "Demanda eléctrica: Serie real y predicciones finales"
)

lines(fc_sarima$mean, col = "red", lwd = 2, lty = 2)
lines(best_dl_fc$mean, col = "blue", lwd = 2)
lines(fc_stl_ets$mean, col = "darkgreen", lwd = 2)

legend(
  "topleft",
  legend = c("Serie real", "SARIMA", best_dl_label, "STL_ETS"),
  col = c("black", "red", "blue", "darkgreen"),
  lty = c(1, 2, 1, 1),
  lwd = 2,
  bty = "n"
)

# =========================================================
# GRÁFICO DEL MEJOR MODELO DL CON FECHAS REALES
# =========================================================

df_real <- data.frame(
  fecha = df$fecha,
  value = as.numeric(y)
)

df_pred <- data.frame(
  fecha = dates_test,
  value = as.numeric(best_dl_fc$mean)
)

colores_dl <- c("Serie real" = "black")
colores_dl[best_dl_label] <- "blue"

p_dl_demanda <- ggplot() +
  geom_line(
    data = df_real,
    aes(x = fecha, y = value, colour = "Serie real"),
    linewidth = 0.8
  ) +
  geom_line(
    data = df_pred,
    aes(x = fecha, y = value, colour = best_dl_label),
    linewidth = 1
  ) +
  scale_colour_manual(
    values = colores_dl
  ) +
  labs(
    title = paste("Predicción del mejor modelo", best_dl_label, "sobre log(demanda eléctrica)"),
    x = "Fecha",
    y = "log(demanda eléctrica)",
    colour = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

print(p_dl_demanda)

# =========================================================
# GRÁFICO ZOOM: ÚLTIMOS 2 MESES
# Mejor modelo DL sobre demanda eléctrica
# Sin usar lubridate
# =========================================================

df_real <- data.frame(
  fecha = as.Date(df$fecha),
  value = as.numeric(y)
)

df_pred <- data.frame(
  fecha = as.Date(dates_test),
  value = as.numeric(best_dl_fc$mean)
)

# Fecha máxima disponible
fecha_max <- max(df_real$fecha, na.rm = TRUE)

# Fecha inicial del zoom: 2 meses naturales antes
fecha_zoom <- seq(fecha_max, by = "-2 months", length.out = 2)[2]

# Filtrar últimos 2 meses
df_real_zoom <- df_real[df_real$fecha >= fecha_zoom, ]

df_pred_zoom <- df_pred[df_pred$fecha >= fecha_zoom, ]

# Colores
colores_dl <- c("Serie real" = "black")
colores_dl[best_dl_label] <- "blue"

# Gráfico con zoom
p_dl_demanda <- ggplot() +
  geom_line(
    data = df_real_zoom,
    aes(x = fecha, y = value, colour = "Serie real"),
    linewidth = 0.8
  ) +
  geom_line(
    data = df_pred_zoom,
    aes(x = fecha, y = value, colour = best_dl_label),
    linewidth = 1
  ) +
  scale_colour_manual(
    values = colores_dl
  ) +
  labs(
    title = paste(
      "Predicción del mejor modelo",
      best_dl_label,
      "sobre log(demanda eléctrica) - Zoom últimos 2 meses"
    ),
    x = "Fecha",
    y = "log(demanda eléctrica)",
    colour = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

print(p_dl_demanda)
# =========================================================
# 7) 4.9.6 ANÁLISIS DE ESTABILIDAD DEL MODELO DL
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
  lookback = 14,
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

comparison_hybrid$Model <- "SARIMA + LSTM"
comparison_hybrid <- comparison_hybrid[, c("Model", "RMSE", "MAE", "MAPE")]

cat("\n================ RESULTADOS HÍBRIDO ================\n")
print(comparison_hybrid)

ranking_hybrid <- rank_models_test(
  list(
    SARIMA = fc_sarima,
    Hybrid = fc_hybrid
  ),
  y_test
)

cat("\n================ SARIMA vs HÍBRIDO ================\n")
print(ranking_hybrid)

p_hybrid <- ggplot(ranking_hybrid, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(RMSE, 4)),
            vjust = -0.3,
            size = 4) +
  labs(
    title = "SARIMA vs SARIMA + LSTM híbrido - Demanda eléctrica",
    x = "Modelo",
    y = "RMSE"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold")) +
  ylim(0, max(ranking_hybrid$RMSE) * 1.1)

print(p_hybrid)



# Fechas reales de toda la serie
fechas_y <- as.Date(df$fecha)

# Serie real en escala log
y_real <- as.numeric(y)

# Predicciones
pred_sarima <- as.numeric(fc_sarima$mean)
pred_hybrid <- as.numeric(fc_hybrid$mean)

# Fechas correspondientes a las predicciones
# Si las predicciones son sobre el conjunto test:
fechas_pred <- as.Date(dates_test)

# Comprobación de longitudes
stopifnot(length(fechas_y) == length(y_real))
stopifnot(length(fechas_pred) == length(pred_sarima))
stopifnot(length(fechas_pred) == length(pred_hybrid))

# Rango del eje Y
ylim_total <- range(
  c(y_real, pred_sarima, pred_hybrid),
  na.rm = TRUE
)

# Gráfico principal
plot(
  fechas_y,
  y_real,
  type = "l",
  col = "black",
  lwd = 2,
  xlab = "Fecha",
  ylab = "log(demanda eléctrica)",
  main = "Demanda eléctrica: SARIMA vs SARIMA + LSTM híbrido",
  ylim = ylim_total
)

# Predicción SARIMA
lines(
  fechas_pred,
  pred_sarima,
  col = "red",
  lwd = 2,
  lty = 2
)

# Predicción híbrida SARIMA + LSTM
lines(
  fechas_pred,
  pred_hybrid,
  col = "blue",
  lwd = 2
)

# Leyenda
legend(
  "topleft",
  legend = c("Serie real", "SARIMA", "SARIMA + LSTM"),
  col = c("black", "red", "blue"),
  lty = c(1, 2, 1),
  lwd = 2,
  bty = "n"
)


# =========================================================
# GRÁFICO ZOOM: ÚLTIMOS 2 MESES
# Demanda eléctrica: SARIMA vs SARIMA + LSTM híbrido
# =========================================================

# Fechas reales de toda la serie
fechas_y <- as.Date(df$fecha)

# Serie real en escala log
y_real <- as.numeric(y)

# Predicciones
pred_sarima <- as.numeric(fc_sarima$mean)
pred_hybrid <- as.numeric(fc_hybrid$mean)

# Fechas correspondientes a las predicciones sobre test
fechas_pred <- as.Date(dates_test)

# Comprobación de longitudes
stopifnot(length(fechas_y) == length(y_real))
stopifnot(length(fechas_pred) == length(pred_sarima))
stopifnot(length(fechas_pred) == length(pred_hybrid))

# =========================================================
# 1) DEFINIR VENTANA DE ZOOM: ÚLTIMOS 2 MESES
# =========================================================

fecha_max <- max(fechas_y, na.rm = TRUE)

# Dos meses naturales antes, usando base R
fecha_zoom <- seq(fecha_max, by = "-2 months", length.out = 2)[2]

# =========================================================
# 2) FILTRAR SERIE REAL Y PREDICCIONES
# =========================================================

idx_real_zoom <- fechas_y >= fecha_zoom
idx_pred_zoom <- fechas_pred >= fecha_zoom

fechas_y_zoom <- fechas_y[idx_real_zoom]
y_real_zoom   <- y_real[idx_real_zoom]

fechas_pred_zoom <- fechas_pred[idx_pred_zoom]
pred_sarima_zoom <- pred_sarima[idx_pred_zoom]
pred_hybrid_zoom <- pred_hybrid[idx_pred_zoom]

# =========================================================
# 3) RANGO DEL EJE Y EN LA VENTANA DE ZOOM
# =========================================================

ylim_zoom <- range(
  c(
    y_real_zoom,
    pred_sarima_zoom,
    pred_hybrid_zoom
  ),
  na.rm = TRUE
)

# =========================================================
# 4) GRÁFICO
# =========================================================

plot(
  fechas_y_zoom,
  y_real_zoom,
  type = "l",
  col = "black",
  lwd = 2,
  xlab = "Fecha",
  ylab = "log(demanda eléctrica)",
  main = "Demanda eléctrica: SARIMA vs SARIMA + LSTM híbrido (zoom últimos 2 meses)",
  ylim = ylim_zoom
)

lines(
  fechas_pred_zoom,
  pred_sarima_zoom,
  col = "red",
  lwd = 2,
  lty = 2
)

lines(
  fechas_pred_zoom,
  pred_hybrid_zoom,
  col = "blue",
  lwd = 2
)

legend(
  "topleft",
  legend = c("Serie real", "SARIMA", "SARIMA + LSTM"),
  col = c("black", "red", "blue"),
  lty = c(1, 2, 1),
  lwd = 2,
  bty = "n"
)

plot(
  resid_train,
  main = "Residuos del modelo SARIMA",
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

cat("\nSARIMA vs BEST_DL\n")
print(dm_sarima_bestdl)

cat("\nSARIMA vs STL_ETS\n")
print(dm_sarima_stl)

cat("\nSARIMA vs HYBRID\n")
print(dm_sarima_hybrid)


# =========================================================
# 10) GRÁFICO FINAL COMPARATIVO
# =========================================================

ranking_base <- ranking_final[, c("Model", "RMSE", "MAE", "MAPE")]

ranking_all <- rbind(
  ranking_base,
  data.frame(
    Model = "SARIMA_LSTM_HYBRID",
    RMSE  = comparison_hybrid$RMSE,
    MAE   = comparison_hybrid$MAE,
    MAPE  = comparison_hybrid$MAPE
  )
)

# =========================================================
# GRUPOS DE COLORES
# =========================================================

ranking_all$Grupo <- "Otros"
ranking_all$Grupo[ranking_all$Model == "LSTM_BEST"] <- "Deep Learning"
ranking_all$Grupo[ranking_all$Model == "STL_ETS"] <- "Mejor clásico"
ranking_all$Grupo[ranking_all$Model == "SARIMA_LSTM_HYBRID"] <- "Híbrido"

# =========================================================
# ORDEN: peores arriba, mejores abajo
# =========================================================

ranking_all <- ranking_all[order(ranking_all$RMSE, decreasing = FALSE), ]

ranking_all$Model <- factor(
  ranking_all$Model,
  levels = ranking_all$Model
)

# =========================================================
# GRÁFICO
# =========================================================

p_all <- ggplot(ranking_all, aes(x = Model, y = RMSE, fill = Grupo)) +
  geom_col() +
  coord_flip() +
  geom_text(
    aes(label = round(RMSE, 4)),
    hjust = -0.15,
    size = 4
  ) +
  scale_fill_manual(
    values = c(
      "Deep Learning" = "blue",
      "Mejor clásico" = "black",
      "Híbrido" = "darkgreen",
      "Otros" = "grey70"
    )
  ) +
  labs(
    title = "Comparación final completa de modelos - Demanda eléctrica",
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