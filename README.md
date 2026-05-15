# TFG - Predicción de series temporales mediante modelos clásicos y Deep Learning

Este repositorio contiene los scripts en R utilizados en el Trabajo de Fin de Grado dedicado a la comparación de modelos clásicos de predicción de series temporales y modelos basados en redes neuronales, principalmente LSTM y GRU.

El objetivo del repositorio es facilitar la transparencia y reproducibilidad del análisis empírico desarrollado en el trabajo, dejando disponible el código empleado para la preparación de las series, la estimación de los modelos, la obtención de predicciones, el cálculo de métricas de error y la generación de resultados.

## Descripción del proyecto

El trabajo analiza distintas series temporales reales con características diversas: series estacionales, series no estacionales, series económicas, series financieras y series de mayor complejidad temporal. La finalidad es comparar el comportamiento predictivo de distintas familias de modelos:

- Modelos benchmark: naive, seasonal naive y drift.
- Modelos clásicos: ARIMA, SARIMA, ETS y STL.
- Modelos de Deep Learning: LSTM y GRU.
- Modelos híbridos, cuando procede, combinando estructuras clásicas y redes neuronales.

La comparación entre modelos se realiza mediante métricas de error como RMSE, MAE y MAPE, prestando especial atención a la capacidad predictiva fuera de muestra.

## Scripts incluidos

Los scripts principales se encuentran en la raíz del repositorio:

| Script | Serie temporal |
|---|---|
| `AIRPASSANGERS.R` | Pasajeros aéreos internacionales |
| `JOHNSHON.R` | Ganancias trimestrales de Johnson & Johnson |
| `CO2.R` | Concentración atmosférica de CO2 |
| `NILE.R` | Caudal anual del río Nilo |
| `Sunspots.R` | Manchas solares |
| `IPC ESPAÑA.R` | IPC de España |
| `CAMBIO USD.EUR.R` | Tipo de cambio USD/EUR |
| `Importaciones italia.R` | Importaciones de bienes de Italia a España |
| `Demanda electrica.R` | Demanda eléctrica |



## Reproducibilidad

Los scripts siguen una lógica común:

1. Carga y preparación de los datos.
2. Análisis exploratorio de la serie.
3. Transformaciones necesarias, cuando proceda.
4. Ajuste de modelos clásicos.
5. Ajuste de modelos LSTM y GRU.
6. Obtención de predicciones fuera de muestra.
7. Cálculo de métricas de error.
8. Comparación de resultados.
9. Generación de gráficos y tablas.



## Software utilizado

El análisis se ha desarrollado principalmente en R. Entre los paquetes utilizados se encuentran:

- `forecast`
- `ggplot2`
- `dplyr`
- `Metrics`
- `keras3`
- `tensorflow`
- `tseries`
- `lubridate`
- `readxl`

## Datos

Algunas series proceden directamente de librerías internas de R, mientras que otras requieren descarga o importación desde fuentes externas.



## Autor

**Pablo Sanz Santiburcio**  
Trabajo de Fin de Grado  
Universidad Carlos III de Madrid

## Finalidad del repositorio

Este repositorio tiene una finalidad exclusivamente académica. Su propósito es documentar y facilitar la reproducibilidad del análisis empírico desarrollado en el Trabajo de Fin de Grado.
