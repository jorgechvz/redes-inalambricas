import numpy as np
import matplotlib.pyplot as plt
import time
from bladerf import _bladerf

def schmidl_cox_algorithm(signal, L, threshold=0.8):
    """
    Aplica el algoritmo de Schmidl-Cox para detectar el preámbulo.

    Parámetros:
    - signal: la señal recibida (array complejo)
    - L: longitud del segmento repetido
    - threshold: umbral para la detección (valor entre 0 y 1)

    Retorna:
    - El índice d donde M(d) es máximo y supera el umbral
    - Los valores de M(d) para graficar
    """
    N = len(signal)
    # Inicializar arrays
    P = np.zeros(N - 2 * L, dtype=complex)
    R = np.zeros(N - 2 * L)
    M = np.zeros(N - 2 * L)

    for d in range(N - 2 * L):
        P[d] = np.sum(signal[d:d+L] * np.conj(signal[d+L:d+2*L]))
        R[d] = np.sum(np.abs(signal[d:d+2*L])**2)
        if R[d] != 0:
            M[d] = np.abs(P[d])**2 / R[d]**2
        else:
            M[d] = 0

    # Normalizar M(d)
    max_M = np.max(M)
    if max_M > 0:
        M = M / max_M
    else:
        M = np.zeros_like(M)

    # Detección del pico que supera el umbral
    peaks = np.where(M > threshold)[0]
    if len(peaks) > 0:
        d_max = peaks[0]
        return d_max, M
    else:
        return None, M

def rrc_filter(beta, sps, num_taps):
    """Genera un filtro de raíz de coseno alzado (RRC)."""
    t = np.arange(-num_taps//2, num_taps//2 + 1) / sps
    pi_t = np.pi * t
    four_beta_t = 4 * beta * t

    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.sin(pi_t * (1 - beta)) + 4 * beta * t * np.cos(pi_t * (1 + beta))
        denominator = pi_t * (1 - (four_beta_t) ** 2)
        h = numerator / denominator

    # Manejar el caso t = 0
    h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    # Manejar el caso |t| = 1 / (4 * beta)
    t_special = np.abs(t) == (1 / (4 * beta))
    h[t_special] = (beta / np.sqrt(2)) * (
        ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) +
        ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    )

    # Normalizar el filtro
    h /= np.sqrt(np.sum(h**2))

    return h

class BladeRFSpectrumPreambleAnalyzer:
    def __init__(self, center_freq, sample_rate, bandwidth, gain, num_samples, preamble_len, samples_per_symbol, beta, num_taps):
        # Parámetros de configuración
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.gain = gain
        self.num_samples = num_samples
        self.preamble_len = preamble_len
        self.samples_per_symbol = samples_per_symbol
        self.beta = beta
        self.num_taps = num_taps

        # Inicializar el BladeRF y el canal de recepción
        self.sdr = _bladerf.BladeRF()
        self.rx_ch = self.sdr.Channel(_bladerf.CHANNEL_RX(0))  # Canal RX
        self.configure_rx_channel()

        # Configurar el stream síncrono
        self.sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X1,
                             fmt=_bladerf.Format.SC16_Q11,
                             num_buffers=16,
                             buffer_size=8192,
                             num_transfers=8,
                             stream_timeout=3500)

    def configure_rx_channel(self):
        """Configura los parámetros del canal de recepción (RX)."""
        self.rx_ch.frequency = self.center_freq  # Frecuencia central
        self.rx_ch.sample_rate = self.sample_rate  # Tasa de muestreo
        self.rx_ch.bandwidth = self.bandwidth  # Ancho de banda
        self.rx_ch.gain_mode = _bladerf.GainMode.Manual  # Modo de ganancia manual
        self.rx_ch.gain = self.gain  # Ganancia en dB

    def receive_samples(self, num_samples):
        """Recibe muestras del BladeRF y las devuelve como un array de números complejos."""
        # Crear un buffer para recibir los datos crudos (bytes)
        bytes_per_sample = 4  # 2 bytes para I y 2 bytes para Q
        buf = bytearray(num_samples * bytes_per_sample)

        # Recibir los datos del bladeRF
        self.sdr.sync_rx(buf, num_samples)

        # Convertir los datos crudos a un array de int16
        samples_int16 = np.frombuffer(buf, dtype=np.int16)

        # Separar los componentes I y Q
        I = samples_int16[0::2]
        Q = samples_int16[1::2]

        # Combinar I y Q en números complejos
        samples = I.astype(np.float32) + 1j * Q.astype(np.float32)

        # Normalizar los valores
        samples /= 2047.0  # Ajusta este factor según sea necesario

        return samples

    def process_preamble(self, samples):
        """Procesa las muestras para detectar el preámbulo usando Schmidl-Cox."""
        # Aplicar el filtro RRC
        rrc_coef = rrc_filter(self.beta, self.samples_per_symbol, self.num_taps)
        samples_filtered = np.convolve(samples, rrc_coef, mode='same')

        # Obtener la longitud L para el algoritmo Schmidl-Cox
        L = (self.preamble_len * self.samples_per_symbol) // 2

        d_max, M = schmidl_cox_algorithm(samples_filtered, L)
        return d_max, M

    def calculate_spectrum(self, samples):
        """Calcula el espectro a partir de las muestras IQ usando FFT."""
        spectrum = np.fft.fftshift(np.fft.fft(samples))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1 / self.sample_rate))
        magnitude = 20 * np.log10(np.abs(spectrum) + 1e-12)
        return freqs, magnitude

    def close(self):
        """Cierra la conexión con el BladeRF."""
        self.sdr.close()

class SpectrumPreamblePlotter:
    def __init__(self, analyzer):
        """Inicializa la clase de visualización y crea los gráficos."""
        self.analyzer = analyzer
        self.init_plot()
        self.preamble_detected = False

    def init_plot(self):
        """Inicializa la ventana de la gráfica interactiva."""
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(15, 10))

        # Configuración del gráfico de espectro
        self.line1, = self.ax1.plot([], [], lw=2)
        self.ax1.set_title(f"Espectro en Tiempo Real - {self.analyzer.center_freq / 1e6} MHz")
        self.ax1.set_xlabel("Frecuencia (MHz)")
        self.ax1.set_ylabel("Magnitud (dB)")
        self.ax1.grid(True)

        # Configuración del gráfico de M(d)
        self.line2, = self.ax2.plot([], [], lw=2, label="M(d)")
        self.ax2.set_title("Detección de Preámbulo usando Schmidl-Cox")
        self.ax2.set_xlabel("Desplazamiento (d)")
        self.ax2.set_ylabel("M(d)")
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Configuración del gráfico en dominio del tiempo
        self.line3, = self.ax3.plot([], [], lw=2, label="Dominio del Tiempo")
        self.ax3.set_title("Señal en Dominio del Tiempo")
        self.ax3.set_xlabel("Muestra")
        self.ax3.set_ylabel("Amplitud")
        self.ax3.grid(True)
        self.ax3.legend()

        plt.tight_layout()

    def update_spectrum_plot(self, freqs, magnitude):
        """Actualiza el gráfico del espectro."""
        self.line1.set_data(freqs / 1e6, magnitude)
        self.ax1.set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
        self.ax1.set_ylim(np.min(magnitude) - 10, np.max(magnitude) + 10)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_preamble_plot(self, M, d_max):
        """Actualiza el gráfico de M(d) y marca el preámbulo detectado."""
        self.line2.set_data(range(len(M)), M)
        self.ax2.set_xlim(0, len(M))
        self.ax2.set_ylim(0, 1.1)
        if d_max is not None:
            print(f"Preámbulo detectado en d = {d_max}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def update_time_domain_plot(self, samples):
        real_part = np.real(samples)  # Asegúrate de usar la parte real de la señal
        num_muestras_a_mostrar = 2000  # Ajusta este valor según cuántas muestras quieres mostrar

        self.line3.set_data(np.arange(num_muestras_a_mostrar), real_part[:num_muestras_a_mostrar])
        self.ax3.set_xlim(0, num_muestras_a_mostrar)  # Cambia el rango a la cantidad de muestras que deseas visualizar
        self.ax3.set_ylim(np.min(real_part[:num_muestras_a_mostrar]) - 0.1, np.max(real_part[:num_muestras_a_mostrar]) + 0.1)  # Ajusta los límites del eje y basado en la porción de la señal que estás graficando
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def start_stream(self):
        self.analyzer.rx_ch.enable = True
        try:
            while True:
                samples = self.analyzer.receive_samples(self.analyzer.num_samples)

                # Actualizar espectro
                freqs, magnitude = self.analyzer.calculate_spectrum(samples)
                self.update_spectrum_plot(freqs, magnitude)

                # Detectar preámbulo
                d_max, M = self.analyzer.process_preamble(samples)
                self.update_preamble_plot(M, d_max)

                # Actualizar dominio del tiempo
                self.update_time_domain_plot(samples)

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Deteniendo recepción.")
        finally:
            self.analyzer.rx_ch.enable = False
            self.analyzer.close()

if __name__ == '__main__':
    center_freq = 940e6
    sample_rate = 5e6
    bandwidth = 10e6
    gain = 10
    num_samples = 65536
    preamble_len = 200
    samples_per_symbol = 8
    beta = 0.2
    num_taps = 151

    analyzer = BladeRFSpectrumPreambleAnalyzer(center_freq, sample_rate, bandwidth, gain,
                                               num_samples, preamble_len, samples_per_symbol,
                                               beta, num_taps)
    plotter = SpectrumPreamblePlotter(analyzer)

    plotter.start_stream()