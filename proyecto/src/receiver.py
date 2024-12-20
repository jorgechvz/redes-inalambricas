import numpy as np
from PIL import Image
from bladerf import _bladerf
from modulation_utils import (
    rrc_filter,
    mueller_muller_timing_recovery,
    costas_loop,
    channel_decode,
    symbols_to_bits,
    schmidl_cox_algorithm_vectorized,
    plot_constellation
)
import matplotlib.pyplot as plt


class BladeRFReceiver:
    def __init__(self, sample_rate=5e6, center_freq=940e6, gain=20):
        self.sdr = _bladerf.BladeRF()
        self.rx_ch = self.sdr.Channel(_bladerf.CHANNEL_RX(0))
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain

        self.rx_ch.frequency = self.center_freq
        self.rx_ch.sample_rate = self.sample_rate
        self.rx_ch.bandwidth = self.sample_rate / 2
        self.rx_ch.gain = self.gain
        self.rx_ch.gain_mode = _bladerf.GainMode.Manual

        self.sdr.sync_config(
            layout=_bladerf.ChannelLayout.RX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=8192,
            num_transfers=8,
            stream_timeout=3500,
        )

        self.rx_ch.enable = True

    def receive_samples(self, num_samples):
        bytes_per_sample = 4
        buf = bytearray(num_samples * bytes_per_sample)
        self.sdr.sync_rx(buf, num_samples)
        samples_int16 = np.frombuffer(buf, dtype=np.int16)
        I = samples_int16[0::2]
        Q = samples_int16[1::2]
        samples = I.astype(np.float32) + 1j * Q.astype(np.float32)
        samples /= 2047.0
        return samples

    def close(self):
        self.rx_ch.enable = False
        self.sdr.close()


def estimate_cfo(
    received_signal, d_max, preamble_symbols, samples_per_symbol, sample_rate
):
    """Estima el Desplazamiento de Frecuencia Portadora (CFO) usando el preámbulo."""
    # Extraer el segmento del preámbulo recibido
    preamble_length = len(preamble_symbols) * samples_per_symbol
    preamble_signal = received_signal[d_max : d_max + preamble_length]

    # Crear el preámbulo conocido sobresampleado
    known_preamble = 2 * preamble_symbols - 1
    known_preamble_upsampled = np.repeat(known_preamble, samples_per_symbol)

    # Estimar el CFO
    product = preamble_signal * np.conj(known_preamble_upsampled)
    angle = np.angle(np.sum(product))
    freq_offset_est = angle / (2 * np.pi * (preamble_length / sample_rate))

    return freq_offset_est


if __name__ == "__main__":
    # Parámetros de recepción
    sample_rate = 5e6
    center_freq = 940e6
    gain = 20
    samples_per_symbol = 4
    beta = 0.35
    num_taps = 101
    repetition = 3

    # Cantidad de muestras a recibir
    preamble_bits = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    preamble_symbols = 2 * preamble_bits - 1
    num_samples = 65536
    L = len(preamble_symbols) * samples_per_symbol

    # Instancia del receptor
    radio_rx = BladeRFReceiver(sample_rate, center_freq, gain)
    try:
        received_signal = radio_rx.receive_samples(num_samples)
        print("Recibiendo muestras...")

        d_max, M = schmidl_cox_algorithm_vectorized(received_signal, L, threshold=0.1)
        freq_offset_est = estimate_cfo(
            received_signal,
            d_max,
            preamble_bits,
            samples_per_symbol,
            sample_rate,
        )

        # Corregir el CFO a partir de d_max
        t = np.arange(len(received_signal[d_max:])) / sample_rate
        received_signal[d_max:] *= np.exp(-1j * 2 * np.pi * freq_offset_est * t)

        # Filtrado adaptado con filtro RRC
        rrc_coef = rrc_filter(beta, samples_per_symbol, num_taps)
        matched_filtered_signal = np.convolve(received_signal, rrc_coef, mode="same")

        # Sincronización en tiempo usando Mueller y Müller
        timing_recovered_signal = mueller_muller_timing_recovery(
            matched_filtered_signal[d_max:], samples_per_symbol
        )

        # Sincronización de frecuencia fina usando Costas Loop
        frequency_corrected_signal = costas_loop(timing_recovered_signal)

        # Demodulación
        received_symbols = frequency_corrected_signal
        received_bits = (np.real(received_symbols) > 0).astype(np.uint8)
        # Extraer los bits del preámbulo recibidos
        preamble_received_bits = received_bits[: len(preamble_bits)]
        # Verificación del preámbulo
        preamble_match = np.array_equal(preamble_received_bits, preamble_bits)
        preamble_inverted_match = np.array_equal(
            1 - preamble_received_bits, preamble_bits
        )
        if preamble_inverted_match:
            print("Se detectó inversión de fase. Corrigiendo inversión de símbolos.")
            # Invertir los símbolos recibidos
            received_bits = 1 - received_bits

        # Ahora aplicar la decodificación de canal al resto de los bits
        # Calcular la longitud esperada del encabezado y de los bits de imagen
        header_length = (
            32 * repetition
        )  # 32 bits del encabezado codificados con repetición
        received_bits_rest = received_bits[len(preamble_bits) :]

        # Decodificar los bits del encabezado
        encoded_header_bits = received_bits_rest[:header_length]

        decoded_header_bits = channel_decode(encoded_header_bits, repetition=repetition)
        width_bits = decoded_header_bits[:16]
        height_bits = decoded_header_bits[16:32]
        modulation_type_bits = decoded_header_bits[32:34]
        image_symbols = received_symbols[len(preamble_bits) :]

        # Convertir bits a enteros de 16 bits
        width_bytes = np.packbits(width_bits)
        height_bytes = np.packbits(height_bits)
        width = int(np.frombuffer(width_bytes.tobytes(), dtype=">u2")[0])
        height = int(np.frombuffer(height_bytes.tobytes(), dtype=">u2")[0])

        # Determinar modulación
        if np.array_equal(modulation_type_bits, [0, 0]):
            modulation_scheme = {"before_processing": received_signal, "after_processing": received_symbols, "modulation_type": "QPSK"}
            data = symbols_to_bits(image_symbols, modulation_scheme["modulation_type"])
        elif np.array_equal(modulation_type_bits, [0, 1]):
            modulation_scheme = {"before_processing": received_signal, "after_processing": received_symbols, "modulation_type": "8QAM"}
            data = symbols_to_bits(image_symbols, modulation_scheme["modulation_type"])
        elif np.array_equal(modulation_type_bits, [1, 0]):
            modulation_scheme = {"before_processing": received_signal, "after_processing": received_symbols, "modulation_type": "16QAM"}
            data = symbols_to_bits(image_symbols, modulation_scheme["modulation_type"])
        else:
            raise ValueError("Modulación desconocida en el encabezado")

        print(f"Ancho: {width}, Alto: {height}, Modulación: {modulation_scheme}")

        # Decodificar bits de imagen
        decoded_image_bits = channel_decode(data, repetition=repetition)

        # Reconstruir la imagen
        total_image_bits = len(decoded_image_bits)
        total_image_pixels = width * height * 3
        if total_image_pixels * 8 != total_image_bits:
            raise ValueError(
                "La cantidad de bits no coincide con el tamaño esperado de la imagen."
            )

        image_bytes = np.packbits(decoded_image_bits)
        image_array = image_bytes.reshape((height, width, 3))
        image_reconstructed = Image.fromarray(image_array.astype(np.uint8))

        # Crear figura con subplots
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Graficar constelación pasando las axes creadas externamente
        plot_constellation(modulation_scheme, ax[0], ax[1])

        # Graficar la imagen en ax[2]
        ax[2].imshow(image_reconstructed)
        ax[2].set_title("Imagen Reconstruida")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        print("Recepción interrumpida por el usuario.")
    finally:
        radio_rx.close()
        pass
