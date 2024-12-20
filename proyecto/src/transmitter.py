# transmitter.py
import numpy as np
from PIL import Image
from bladerf import _bladerf
import time
from modulation_utils import channel_encode, rrc_filter, bits_to_symbols


class BladeRFRadio:
    def __init__(self, sample_rate=5e6, center_freq=940e6, gain=60):
        self.sdr = _bladerf.BladeRF()
        self.tx_ch = self.sdr.Channel(_bladerf.CHANNEL_TX(0))
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain

        self.tx_ch.frequency = self.center_freq
        self.tx_ch.sample_rate = self.sample_rate
        self.tx_ch.bandwidth = self.sample_rate / 2
        self.tx_ch.gain = self.gain

    def prepare_signal_for_transmission(self, signal):
        scale_factor = 2047
        signal_i = np.real(signal) * scale_factor
        signal_q = np.imag(signal) * scale_factor
        signal_i = signal_i.astype(np.int16)
        signal_q = signal_q.astype(np.int16)
        interleaved = np.empty((signal_i.size + signal_q.size,), dtype=np.int16)
        interleaved[0::2] = signal_i
        interleaved[1::2] = signal_q
        buf = interleaved.tobytes()
        return buf

    def start_transmit(self, signal, duration_sec):
        buf = self.prepare_signal_for_transmission(signal)
        num_samples = len(signal)

        self.sdr.sync_config(
            layout=_bladerf.ChannelLayout.TX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=8192,
            num_transfers=8,
            stream_timeout=3500,
        )

        print("Iniciando transmisión...")
        self.tx_ch.enable = True

        start_time = time.time()
        while time.time() - start_time < duration_sec:
            self.sdr.sync_tx(buf, num_samples)

        print("Deteniendo transmisión")
        self.tx_ch.enable = False
        self.sdr.close()


if __name__ == "__main__":
    modulation_scheme_data = input(
        "Ingrese el tipo de modulación para los datos (QPSK, 8QAM, 16QAM): "
    )

    # Configuración de parámetros
    image_path = "../images/example.jpg"
    sample_rate = 1e6
    center_freq = 940e6
    gain = 60
    samples_per_symbol = 4
    duration_sec = 10
    beta = 0.35
    num_taps = 101
    repetition = 3

    # PREÁMBULO
    preamble_bits = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1])  # Preambulo
    preamble_symbols = 2 * preamble_bits - 1  # Símbolos BPSK

    # Cargar imagen y convertir a bits
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_array = np.array(image)
    image_flat = image_array.flatten()
    image_bits = np.unpackbits(image_flat)  # Bits de la imagen

    print(f"Imagen cargada: {width}x{height}, Total bits: {len(image_bits)}")

    # Codificar bits de la imagen
    encoded_bits = channel_encode(image_bits, repetition=repetition)

    # Crear encabezado
    width_bytes = np.array([width], dtype=">u2").view(np.uint8)
    height_bytes = np.array([height], dtype=">u2").view(np.uint8)
    width_bits = np.unpackbits(width_bytes)
    height_bits = np.unpackbits(height_bytes)

    modulation_map = {"QPSK": [0, 0], "8QAM": [0, 1], "16QAM": [1, 0]}
    modulation_bits = np.array(modulation_map[modulation_scheme_data], dtype=np.uint8)
    header_bits = np.concatenate((width_bits, height_bits, modulation_bits))

    encoded_header_bits = channel_encode(header_bits, repetition=repetition)

    # Modulación
    header_symbols = bits_to_symbols(encoded_header_bits, "BPSK")
    data_symbols = bits_to_symbols(encoded_bits, modulation_scheme_data)
    all_symbols = np.concatenate((preamble_symbols, header_symbols, data_symbols))

    # Sobremuestrear y filtrar
    symbols_upsampled = np.zeros(len(all_symbols) * samples_per_symbol, dtype=complex)
    symbols_upsampled[::samples_per_symbol] = all_symbols

    rrc_coef = rrc_filter(beta, samples_per_symbol, num_taps)
    signal_filtered = np.convolve(symbols_upsampled, rrc_coef, mode="same")
    signal_filtered /= np.max(np.abs(signal_filtered))
    signal_complex = signal_filtered.astype(np.complex64)
    print("Señal lista para transmitir.", signal_complex)
    print("Iniciando transmision...")
    # Transmitir señal
    radio_tx = BladeRFRadio(sample_rate, center_freq, gain)
    radio_tx.start_transmit(signal_complex, duration_sec=duration_sec) 
