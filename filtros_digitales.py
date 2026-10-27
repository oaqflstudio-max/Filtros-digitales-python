import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================
# 1. DEFINICIÓN DE LA SEÑAL DE ENTRADA
# ==========================================
fs = 1000  # Frecuencia de muestreo (Hz)
t = np.linspace(0, 1.0, fs, endpoint=False)  # Tiempo de 1 segundo

# Señal compuesta: suma de onda de baja (5 Hz), media (50 Hz) y alta frecuencia (150 Hz)
sig_5hz = np.sin(2 * np.np.pi if hasattr(np, 'np') else 2 * np.pi * 5 * t)
sig_50hz = 0.5 * np.sin(2 * np.pi * 50 * t)
sig_150hz = 0.3 * np.sin(2 * np.pi * 150 * t)

# Ruido blanco gaussiano
np.random.seed(42)  # Semilla para reproducibilidad
noise = np.random.normal(0, 0.4, t.shape)

# Señal de entrada limpia y con ruido
signal_clean = sig_5hz + sig_50hz + sig_150hz
signal_noisy = signal_clean + noise

# ==========================================
# 2. DISEÑO DE FILTROS DIGITALES
# ==========================================
nyquist = 0.5 * fs

# a) Filtro Pasa Bajos (Butterworth IIR) - Corta frecuencias mayores a 15 Hz
cutoff_low = 15.0
b_low, a_low = signal.butter(N=4, Wn=cutoff_low / nyquist, btype='lowpass')

# b) Filtro Pasa Altos (Chebyshev Tipo I IIR) - Mantiene frecuencias superiores a 100 Hz
cutoff_high = 100.0
b_high, a_high = signal.cheby1(N=4, rp=1, Wn=cutoff_high / nyquist, btype='highpass')

# c) Filtro Pasa Bandas (FIR con ventana Hamming) - Aísla la frecuencia de 50 Hz (30-70 Hz)
numtaps = 101
band = [30.0 / nyquist, 70.0 / nyquist]
b_band = signal.firwin(numtaps, band, pass_zero=False, window='hamming')
a_band = [1.0]

# ==========================================
# 3. APLICACIÓN DE LOS FILTROS
# ==========================================
# Filtrado de la señal ruidosa
filtered_low = signal.filtfilt(b_low, a_low, signal_noisy)
filtered_high = signal.filtfilt(b_high, a_high, signal_noisy)
filtered_band = signal.lfilter(b_band, a_band, signal_noisy)

# ==========================================
# 4. VISUALIZACIÓN DE RESULTADOS
# ==========================================

# --- GRÁFICA 1: Dominio del Tiempo ---
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t, signal_noisy, label='Señal con Ruido (Entrada)', color='gray', alpha=0.7)
plt.plot(t, signal_clean, label='Señal Limpia Original', color='black', linestyle='--')
plt.title('Señal de Entrada (Tiempo)')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, filtered_low, label='Filtro Pasa Bajos (Butterworth)', color='blue')
plt.plot(t, sig_5hz, label='Componente deseada (5 Hz)', color='green', linestyle=':')
plt.title('Resultado: Filtro Pasa Bajos (< 15 Hz)')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, filtered_high, label='Filtro Pasa Altos (Chebyshev)', color='red')
plt.plot(t, sig_150hz, label='Componente deseada (150 Hz)', color='orange', linestyle=':')
plt.title('Resultado: Filtro Pasa Altos (> 100 Hz)')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, filtered_band, label='Filtro Pasa Bandas (FIR Window)', color='purple')
plt.plot(t, sig_50hz, label='Componente deseada (50 Hz)', color='magenta', linestyle=':')
plt.title('Resultado: Filtro Pasa Bandas (30-70 Hz)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- GRÁFICA 2: Respuesta en Frecuencia (Espectro FFT) ---
def plot_fft(sig, fs, label, color):
    n = len(sig)
    freq = np.fft.rfftfreq(n, d=1/fs)
    magnitude = np.abs(np.fft.rfft(sig)) / n
    plt.plot(freq, magnitude, label=label, color=color)

plt.figure(figsize=(10, 6))
plot_fft(signal_noisy, fs, 'Señal Entrada con Ruido', 'gray')
plot_fft(filtered_low, fs, 'Pasa Bajos (<15 Hz)', 'blue')
plot_fft(filtered_band, fs, 'Pasa Bandas (30-70 Hz)', 'purple')
plot_fft(filtered_high, fs, 'Pasa Altos (>100 Hz)', 'red')

plt.title('Respuesta y Análisis en el Dominio de la Frecuencia (FFT)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.xlim(0, 200)
plt.legend()
plt.grid(True)
plt.show()
