import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import time
import random

# Parámetros de simulación
CENTER_FREQ = 107.7e6
SAMPLE_RATE = 10e6
NUM_SAMPLES = int(1e3)

class MockBladeRF:
    """Clase que simula un dispositivo BladeRF generando datos de espectro."""
    
    def __init__(self, center_freq=CENTER_FREQ, sample_rate=SAMPLE_RATE, gain=50, bandwidth=5e6, noise_floor=-90, max_signal=-20):
        self.frequency = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.bandwidth = bandwidth
        self.noise_floor = noise_floor  # dBm
        self.max_signal = max_signal    # dBm

    def generate_mock_spectrum(self):
        # Generar ruido base
        noise = np.random.normal(0, 0.1, NUM_SAMPLES)

        # Simular algunas señales
        t = np.arange(NUM_SAMPLES) / self.sample_rate

        # Señales fijas
        signal1 = 0.5 * np.sin(2 * np.pi * 1e6 * t)  # Señal a 1 MHz del centro
        signal2 = 0.3 * np.sin(2 * np.pi * 2e6 * t)  # Señal a 2 MHz del centro

        # Señales aleatorias
        num_random_signals = random.randint(1, 5)
        random_signals = np.zeros_like(noise)
        for _ in range(num_random_signals):
            freq = random.uniform(-self.bandwidth / 2, self.bandwidth / 2)
            amplitude = random.uniform(0.1, 0.4)
            random_signals += amplitude * np.sin(2 * np.pi * freq * t)

        # Combinar señales y ruido
        combined_signal = noise + signal1 + signal2 + random_signals

        # Aplicar FFT
        spectrum = np.fft.fft(combined_signal)
        spectrum = np.abs(spectrum[:NUM_SAMPLES // 2])  # Solo mantener la mitad positiva

        # Convertir a dBm
        spectrum_dbm = 10 * np.log10(spectrum**2 + 1e-10) + self.noise_floor

        # Aplicar límites realistas
        spectrum_dbm = np.clip(spectrum_dbm, self.noise_floor, self.max_signal)

        # Añadir fluctuaciones aleatorias
        spectrum_dbm += np.random.normal(0, 1, spectrum_dbm.shape)

        freqs = np.fft.fftfreq(NUM_SAMPLES, 1 / self.sample_rate)[:NUM_SAMPLES // 2]

        return freqs, spectrum_dbm

class SpectrumPlot:
    """Clase para visualizar el espectro en tiempo real usando matplotlib."""
    
    def __init__(self):
        self.mock_sdr = MockBladeRF()
        self.init_plots()

    def init_plots(self):
        # Crear la figura y los ejes
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title("Espectro en Tiempo Real")
        self.ax.set_xlabel("Frecuencia (MHz)")
        self.ax.set_ylabel("Potencia (dBm)")
        
        # Inicializar la gráfica
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_xlim(105, 110)  # Rango de frecuencia en MHz
        self.ax.set_ylim(-100, -20)  # Rango de potencia en dBm
        
        # Mostrar la figura sin bloquear la ejecución
        plt.show(block=False)

    def update_plot(self):
        freqs, spectrum = self.mock_sdr.generate_mock_spectrum()
        freqs_mhz = freqs / 1e6  # Convertir a MHz

        # Actualizar los datos del gráfico
        self.line.set_xdata(freqs_mhz)
        self.line.set_ydata(spectrum)

        # Ajustar los límites de los ejes si es necesario
        self.ax.set_xlim(freqs_mhz.min(), freqs_mhz.max())
        
        # Redibujar la figura
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def start_stream(self):
        print("Iniciando stream de datos...")
        while True:
            self.update_plot()
            time.sleep(0.1)  # Actualizar cada 100 ms

if __name__ == '__main__':
    plotter = SpectrumPlot()
    plotter.start_stream()
