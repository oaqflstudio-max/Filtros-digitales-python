# Actividad Formativa 3: Implementación y Evaluación de Filtros Digitales

## Descripción del Proyecto
Este proyecto analiza e implementa filtros digitales (pasa bajos, pasa altos y pasa bandas) utilizando Python (scipy.signal y matplotlib). Se evalúa su rendimiento en la atenuación de ruido blanco gaussiano y aislamiento de frecuencias específicas.

## Estructura del Código
- *Definición de Entrada*: Generación de una señal sintética de 5 Hz, 50 Hz y 150 Hz sumada a ruido blanco.
- *Diseño de Filtros*:
  - Pasa Bajos: Butterworth IIR de orden 4 con fc = 15 Hz.
  - Pasa Altos: Chebyshev Tipo I IIR de orden 4 con fc = 100 Hz.
  - Pasa Bandas: FIR de 101 coeficientes con ventana Hamming para la banda de 30 - 70 Hz.
- *Resultados*: Comparación temporal y frecuencial de las señales mediante Transformada Rápida de Fourier (FFT).

## Requisitos de Instalación
```bash
pip install numpy scipy matplotlib
