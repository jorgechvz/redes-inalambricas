import numpy as np
from PIL import Image
from bladerf import _bladerf
import time
from utils import rrc_filter, channel_encode

class BladeRFRadio:
    def __init__(self, sample_rate=5e6, center_freq=940e6, gain=60):
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
        self.tx_ch.bandwidth = self.sample_rate / 2  # Configura el ancho de banda
        self.tx_ch.gain = self.gain  # Configura la ganancia

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

    def start_transmit(self, signal, duration_sec):
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
        self.sdr.close()

if __name__ == "__main__":
    # Parámetros de configuración
    sample_rate = 5e6  # Tasa de muestreo
    center_freq = 940e6  # Frecuencia central
    gain = 60  # Ganancia
    samples_per_symbol = 8  # Número de muestras por símbolo
    duration_sec = 10  # Duración de la transmisión en segundos
    beta = 0.2  # Factor de rodadura del filtro RRC
    num_taps = 101  # Número de coeficientes del filtro RRC

    # Generar preámbulo
    preamble_bits = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    preamble_symbols = 2 * preamble_bits - 1  # Mapear bits a símbolos BPSK (-1, +1)

    # Codificación de canal (repetición)
    repetition = 3

    # Cargar imagen y convertir a bits
    image = Image.open('../images/Escudo_UNSA.png')
    image = image.convert('RGB')  # Mantener la imagen en color RGB
    width, height = image.size
    image_array = np.array(image)
    image_flat = image_array.flatten()
    image_bits = np.unpackbits(image_flat)
    print(f"Imagen cargada: {width}x{height} píxeles.")

    # Codificar los bits de imagen
    encoded_bits = channel_encode(image_bits, repetition=repetition)

    # Crear encabezado con el tamaño de la imagen
    width_bytes = np.array([width], dtype='>u2').view(np.uint8)
    height_bytes = np.array([height], dtype='>u2').view(np.uint8)
    width_bits = np.unpackbits(width_bytes)
    height_bits = np.unpackbits(height_bytes)
    header_bits = np.concatenate((width_bits, height_bits))

    # Codificar el encabezado
    encoded_header_bits = channel_encode(header_bits, repetition=repetition)

    # Concatenar preámbulo, encabezado y bits de imagen
    bits = np.concatenate((preamble_bits, encoded_header_bits, encoded_bits))

    # Guardar bits transmitidos para comparación en el receptor
    np.save('../data/transmitted_bits.npy', bits)
    np.save('../data/preamble_bits.npy', preamble_bits)

    # Mapear bits a símbolos BPSK (-1, +1)
    symbols = 2 * bits - 1

    # Sobremuestrear símbolos
    symbols_upsampled = np.zeros(len(symbols) * samples_per_symbol)
    symbols_upsampled[::samples_per_symbol] = symbols

    # Generar filtro RRC
    rrc_coef = rrc_filter(beta, samples_per_symbol, num_taps)

    # Filtrar la señal
    signal_filtered = np.convolve(symbols_upsampled, rrc_coef, mode='same')

    # Normalizar amplitud de la señal
    signal_filtered /= np.max(np.abs(signal_filtered))

    # Convertir a formato complejo (BladeRF espera muestras complejas)
    signal_complex = signal_filtered.astype(np.complex64)

    # Instanciar el radio y transmitir la señal
    radio_tx = BladeRFRadio(sample_rate, center_freq, gain)
    radio_tx.start_transmit(signal_complex, duration_sec=duration_sec)
