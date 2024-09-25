import numpy as np
import matplotlib.pyplot as plt
import time
from bladerf import _bladerf
from scipy.fftpack import fft, fftshift, fftfreq

class BladeRFSpectrumAnalyzer:
    def __init__(self, center_freq, sample_rate, bandwidth, gain, num_samples):
        # Almacenar los parámetros de configuración
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.gain = gain
        self.num_samples = num_samples
        
        # Inicializar el BladeRF y el canal de recepción
        self.sdr = _bladerf.BladeRF()
        self.rx_ch = self.sdr.Channel(_bladerf.CHANNEL_RX(0))  # Configura el canal RX (recepción)
        self.configure_rx_channel()  # Configura los parámetros del canal de recepción
        
        # Configurar el stream síncrono para recibir datos IQ
        self.sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X1,  # Canal RX único
                             fmt=_bladerf.Format.SC16_Q11,         # Formato de 16-bit signed integer
                             num_buffers=16,  # Número de buffers en la transmisión
                             buffer_size=8192,  # Tamaño del buffer
                             num_transfers=8,   # Número de transferencias concurrentes
                             stream_timeout=3500)  # Tiempo de espera para el stream en milisegundos
        
    def configure_rx_channel(self):
        """Configura los parámetros del canal de recepción (RX)."""
        self.rx_ch.frequency = self.center_freq  # Establecer la frecuencia central
        self.rx_ch.sample_rate = self.sample_rate  # Establecer la tasa de muestreo
        self.rx_ch.bandwidth = self.bandwidth  # Establecer el ancho de banda del canal
        self.rx_ch.gain_mode = _bladerf.GainMode.Manual  # Establecer el modo de ganancia manual
        self.rx_ch.gain = self.gain  # Establecer la ganancia en dB
        
    def receive_samples(self):
        """Recibe muestras del BladeRF y las devuelve como un array de números complejos."""
        bytes_per_sample = 4  # 2 bytes para la parte real (I) y 2 bytes para la imaginaria (Q)
        buf = bytearray(1024 * bytes_per_sample)  # Crear un buffer para recibir los datos IQ
        x = np.zeros(self.num_samples, dtype=np.complex64)  # Array para almacenar las muestras complejas
        num_samples_read = 0  # Contador de muestras leídas
        
        # Recibir las muestras IQ hasta completar el número requerido
        while num_samples_read < self.num_samples:
            num = min(len(buf) // bytes_per_sample, self.num_samples - num_samples_read)
            self.sdr.sync_rx(buf, num)  # Recibir muestras y almacenarlas en el buffer
            samples = np.frombuffer(buf, dtype=np.int16)  # Convertir el buffer a enteros de 16 bits
            iq_samples = samples[0::2] + 1j * samples[1::2]  # Combinar las partes I y Q en muestras complejas
            iq_samples /= 2048.0  # Normalizar las muestras
            x[num_samples_read:num_samples_read + num] = iq_samples[:num]  # Almacenar las muestras en el array
            num_samples_read += num  # Actualizar el contador de muestras leídas
            
        return x
    
    def calculate_spectrum(self, samples):
        """Calcula el espectro a partir de las muestras IQ usando FFT."""
        spectrum = fftshift(fft(samples))  # Aplicar la FFT y centrar la frecuencia
        freqs = fftshift(fftfreq(self.num_samples, 1 / self.sample_rate))  # Generar las frecuencias correspondientes
        magnitude = 20 * np.log10(np.abs(spectrum) + 1e-12)  # Calcular la magnitud en dB
        return freqs, magnitude  # Devolver las frecuencias y las magnitudes


class SpectrumPlotter:
    def __init__(self, analyzer):
        """Inicializa la clase de visualización y crea el gráfico."""
        self.analyzer = analyzer  # Asociar el analizador de espectro
        self.init_plot()  # Inicializar la configuración de la gráfica
    
    def init_plot(self):
        """Inicializa la ventana de la gráfica interactiva."""
        plt.ion()  # Activar modo interactivo para actualizaciones en tiempo real
        self.fig, self.ax = plt.subplots(figsize=(10, 6))  # Crear figura y ejes de la gráfica
        self.line, = self.ax.plot([], [], lw=2)  # Crear una línea vacía para actualizar con datos
        
        # Configurar los títulos y etiquetas de los ejes
        self.ax.set_title(f"Analizador de Espectro en Tiempo Real con frecuencia central de: {self.analyzer.center_freq / 1e6} MHz")
        self.ax.set_xlabel("Span (MHz)")  # Etiqueta del eje x en MHz
        self.ax.set_ylabel("Magnitud (dB)")  # Etiqueta del eje y en dB
        self.ax.grid()  # Activar el grid en la gráfica

        # Limitar el rango de frecuencia y magnitud
        self.ax.set_xlim(-self.analyzer.sample_rate / 2 / 1e6, self.analyzer.sample_rate / 2 / 1e6)  # Limitar el eje x
        self.ax.set_ylim(-40, 80)  # Limitar el eje y
    
    def update_plot(self, freqs, spectrum):
        """Actualiza el gráfico con los nuevos datos de frecuencia y espectro."""
        self.line.set_xdata(freqs / 1e6)  # Convertir frecuencias a MHz para la gráfica
        self.line.set_ydata(spectrum)  # Actualizar las magnitudes en dB
        
        # Redibujar la gráfica con los nuevos datos
        self.ax.relim()  # Recalcular los límites de los ejes
        self.ax.autoscale_view()  # Ajustar automáticamente la escala de los ejes
        self.fig.canvas.draw()  # Dibujar el gráfico
        self.fig.canvas.flush_events()  # Actualizar eventos en el modo interactivo
    
    def start_stream(self):
        """Inicia el stream y actualiza el gráfico en tiempo real."""
        self.analyzer.rx_ch.enable = True  # Habilitar el canal de recepción
        print("Recepción habilitada.")

        try:
            # Bucle infinito para recibir muestras y actualizar el gráfico
            while True:
                samples = self.analyzer.receive_samples()  # Recibir muestras
                freqs, spectrum = self.analyzer.calculate_spectrum(samples)  # Calcular el espectro
                self.update_plot(freqs, spectrum)  # Actualizar la gráfica
                time.sleep(0.1)  # Pausa para evitar sobrecarga de CPU
        except KeyboardInterrupt:
            # Finalizar el stream si el usuario interrumpe el proceso
            print("Analizador de espectro detenido.")
        finally:
            self.analyzer.rx_ch.enable = False  # Deshabilitar el canal de recepción
            print("Recepción deshabilitada.")


if __name__ == '__main__':
    # Parámetros de configuración del analizador de espectro
    center_freq = 90.3e6     # Frecuencia central: 100 MHz
    sample_rate = 10e6      # Tasa de muestreo: 10 MHz
    bandwidth = 10e6         # Ancho de banda: 5 MHz
    gain = 50               # Ganancia: 50 dB
    num_samples = int(1e3)  # Número de muestras a procesar: 1000

    # Crear instancia del analizador y del visualizador de espectro
    analyzer = BladeRFSpectrumAnalyzer(center_freq, sample_rate, bandwidth, gain, num_samples)
    plotter = SpectrumPlotter(analyzer)

    # Iniciar la visualización del espectro en tiempo real
    plotter.start_stream()
