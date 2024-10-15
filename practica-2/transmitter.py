from matplotlib import pyplot as plt
import numpy as np
from bladerf import _bladerf
import time

def rrc_filter(beta, sps, num_taps):
    """Genera un filtro de raíz de coseno alzado (RRC).

    Args:
        beta (float): Factor de rodadura (0 < beta <= 1).
        sps (int): Número de muestras por símbolo.
        num_taps (int): Número total de coeficientes del filtro.

    Returns:
        np.array: Coeficientes del filtro RRC.
    """
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

class BladeRFRadio:
    def __init__(self, sample_rate=10e6, center_freq=930e6, gain=0):
        # Inicializar el dispositivo bladeRF
        self.sdr = _bladerf.BladeRF()
        # Configurar el canal de transmisión TX
        self.tx_ch = self.sdr.Channel(_bladerf.CHANNEL_TX(0))
        self.sample_rate = sample_rate  # Tasa de muestreo
        self.center_freq = center_freq  # Frecuencia central
        self.gain = gain  # Ganancia de transmisión

        # Configuración del canal de transmisión
        self.tx_ch.frequency = self.center_freq  # Ajusta la frecuencia central
        self.tx_ch.sample_rate = self.sample_rate  # Ajusta la tasa de muestreo
        self.tx_ch.bandwidth = self.sample_rate / 2  # Configura el ancho de banda (la mitad de la tasa de muestreo)
        self.tx_ch.gain = self.gain  # Configura la ganancia

    def set_sample_rate(self, sample_rate_hz):
        # Ajustar la tasa de muestreo
        self.sample_rate = sample_rate_hz
        self.tx_ch.sample_rate = self.sample_rate

    def set_center_frequency(self, freq_hz):
        # Ajustar la frecuencia central
        self.center_freq = freq_hz
        self.tx_ch.frequency = self.center_freq

    def set_gain(self, gain_db):
        # Ajustar la ganancia
        self.gain = gain_db
        self.tx_ch.gain = self.gain

    def prepare_signal_for_transmission(self, signal):
        """Prepara la señal IQ para la transmisión."""
        # Escalar la señal a formato Q11 (-2048 a +2047)
        scale_factor = 2047
        signal_i = np.real(signal) * scale_factor
        signal_q = np.imag(signal) * scale_factor
        # Convertir a enteros de 16 bits
        signal_i = signal_i.astype(np.int16)
        signal_q = signal_q.astype(np.int16)
        # Intercalar muestras I y Q
        interleaved = np.empty((signal_i.size + signal_q.size,), dtype=np.int16)
        interleaved[0::2] = signal_i
        interleaved[1::2] = signal_q
        buf = interleaved.tobytes()
        return buf

    def start_transmit_with_preamble(self, signal, duration_sec):
        """Configura y comienza la transmisión de la señal."""
        buf = self.prepare_signal_for_transmission(signal)  # Prepara la señal para transmisión
        num_samples = len(signal)

        # Configurar el stream síncrono para la transmisión
        self.sdr.sync_config(layout=_bladerf.ChannelLayout.TX_X1,
                             fmt=_bladerf.Format.SC16_Q11,  # Formato de datos int16
                             num_buffers=16,
                             buffer_size=8192,
                             num_transfers=8,
                             stream_timeout=3500)

        # Iniciar la transmisión
        print("Iniciando transmisión...")
        self.tx_ch.enable = True  # Habilitar el canal de transmisión

        start_time = time.time()
        # Transmitir la señal durante el tiempo especificado
        while time.time() - start_time < duration_sec:
            self.sdr.sync_tx(buf, num_samples)  # Escribir el buffer a BladeRF

        # Finalizar la transmisión
        print("Deteniendo transmisión")
        self.tx_ch.enable = False  # Deshabilitar el canal de transmisión


if __name__ == "__main__":
    # Parámetros de configuración
    sample_rate = 5e6  # Tasa de muestreo: 10 MHz
    center_freq = 940e6  # Frecuencia central: 940 MHz
    gain = 60  # Ganancia: 15 dB
    samples_per_symbol = 8  # Número de muestras por símbolo
    duration_sec = 40  # Duración de la transmisión en segundos
    beta = 0.2  # Factor de rodadura del filtro RRC
    num_taps = 151  # Número de coeficientes del filtro RRC

    # Generación del preámbulo
    preamble_bits = np.array([0, 1] * 100)  # Secuencia alternada de 0 y 1, longitud 200 bits
    preamble_symbols = 2 * preamble_bits - 1  # Mapear bits a símbolos BPSK (-1, +1)

    # Generación de datos aleatorios
    data_bits = np.random.randint(0, 2, size=1000)  # 1000 bits de datos
    data_symbols = 2 * data_bits - 1

    # Concatenar preámbulo y datos
    symbols = np.concatenate((preamble_symbols, data_symbols))

    # Sobremuestreo de los símbolos
    symbols_upsampled = np.zeros(len(symbols) * samples_per_symbol)
    symbols_upsampled[::samples_per_symbol] = symbols  # Insertar símbolos con ceros entre ellos

    # Generar el filtro RRC
    rrc_coef = rrc_filter(beta, samples_per_symbol, num_taps)

    # Filtrar la señal
    signal_filtered = np.convolve(symbols_upsampled, rrc_coef, mode='same')

    # Normalizar la amplitud de la señal
    signal_filtered /= np.max(np.abs(signal_filtered))

    # Convertir a formato complejo (BladeRF espera muestras complejas)
    signal_complex = signal_filtered.astype(np.complex64)
    
    np.save('transmitted_signal.npy', signal_complex)
    # Instanciar el radio y transmitir la señal
    radio_tx = BladeRFRadio(sample_rate, center_freq, gain)
    radio_tx.start_transmit_with_preamble(signal_complex, duration_sec=duration_sec)
