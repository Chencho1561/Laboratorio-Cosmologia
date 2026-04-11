# Análisis modelo CDM con Supernovas tipo SNIa 🌌

Este repositorio contiene un script de Python diseñado para realizar inferencia estadística sobre parámetros cosmológicos utilizando datos reales del catálogo **Pantheon+SH0ES**. El enfoque principal es determinar la ecuación de estado de la energía oscura y su posible evolución temporal.

## 🚀 Características

- **Carga de Datos:** Manejo de datos observacionales de redshift ($z$) y magnitud aparente ($\mu$), incluyendo la lectura de matrices de covarianza comprimidas (`.gz`).
- **Cálculo de Distancias:** Implementación numérica de la distancia por luminosidad mediante integración de la función de Hubble $H(z)$ para un modelo cosmológico con parametrización CPL (Chevallier-Polarski-Linder):
  
  $$H(z) = H_0 \sqrt{\Omega_m(1+z)^3 + \Omega_{DE}(1+z)^{3(1+w_0+w_a)} e^{-3w_a \frac{z}{1+z}}}$$
- **Optimización:** Uso de `scipy.optimize` para la minimización de $\chi^2$.
- **Intervalos de Confianza:** Cálculo de errores a $1\sigma$, $2\sigma$ y $3\sigma$.
- **Gráficos Interactivos:** Generación de contornos de confianza con punteros interactivos para explorar el espacio de parámetros.

## 🛠️ Requisitos

Asegúrate de tener instaladas las siguientes librerías:

```bash
pip install numpy matplotlib scipy
conda install numpy matplotlib scipy
```
## 📀 Ejecutables/Releases
Se adjuntan dos versiones (una para MacOS y otra para Windows) del script.
De esta forma no es necesariio utilizar python ni ninguna de sus librerias directamente
