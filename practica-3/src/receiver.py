import numpy as np
from PIL import Image
from bladerf import _bladerf
from src.utils import (
    rrc_filter,
    schmidl_cox_algorithm_vectorized,
    mueller_muller_timing_recovery,
    costas_loop,
    channel_decode,
)


class BladeRFReceiver:
    def __init__(self, sample_rate=5e6, center_freq=940e6, gain=20):
        # Inicializar el dispositivo bladeRF
        self.sdr = _bladerf.BladeRF()
        # Configurar el canal de recepción RX
        self.rx_ch = self.sdr.Channel(_bladerf.CHANNEL_RX(0))
        self.sample_rate = sample_rate  # Tasa de muestreo
        self.center_freq = center_freq  # Frecuencia central
        self.gain = gain  # Ganancia de recepción

        # Configuración del canal de recepción
        self.rx_ch.frequency = self.center_freq  # Ajusta la frecuencia central
        self.rx_ch.sample_rate = self.sample_rate  # Ajusta la tasa de muestreo
        self.rx_ch.bandwidth = self.sample_rate / 2  # Configura el ancho de banda
        self.rx_ch.gain = self.gain  # Configura la ganancia
        self.rx_ch.gain_mode = _bladerf.GainMode.Manual

        # Configurar el stream síncrono para la recepción
        self.sdr.sync_config(
            layout=_bladerf.ChannelLayout.RX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=8192,
            num_transfers=8,
            stream_timeout=3500,
        )

        # Habilitar el canal de recepción
        self.rx_ch.enable = True

    def receive_samples(self, num_samples):
        """Recibe muestras del BladeRF y las devuelve como un array de números complejos."""
        # Crear un buffer para recibir los datos crudos (bytes)
        bytes_per_sample = 4  # 2 bytes para I y 2 bytes para Q
        buf = bytearray(num_samples * bytes_per_sample)

        # Recibir los datos del BladeRF
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

    def close(self):
        """Cierra el dispositivo BladeRF correctamente."""
        self.rx_ch.enable = False  # Deshabilitar el canal de recepción
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
    # Parámetros de configuración
    sample_rate = 5e6  # Tasa de muestreo
    center_freq = 940e6  # Frecuencia central
    gain = 20  # Ganancia
    samples_per_symbol = 8  # Número de muestras por símbolo
    beta = 0.35  # Factor de rodadura del filtro RRC
    num_taps = 101  # Número de coeficientes del filtro RRC
    repetition = 3  # Repetición en codificación de canal

    # Instanciar el receptor
    radio_rx = BladeRFReceiver(sample_rate, center_freq, gain)

    # Número de muestras a recibir en cada iteración
    num_samples = 65536

    # Definir los bits del preámbulo directamente en el código
    preamble_bits = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    preamble_symbols = 2 * preamble_bits - 1
    L = len(preamble_symbols) * samples_per_symbol

    # Umbral ajustado para evitar detecciones falsas
    preamble_threshold = 0.95

    try:
        while True:  # Bucle infinito para mantener el receptor siempre activo
            # Recibir muestras
            received_signal = radio_rx.receive_samples(num_samples)
            received_signal *= -1  # Invertir la señal si es necesario

            # Calcular la energía de la señal recibida
            energy = np.mean(np.abs(received_signal) ** 2)
            energy_threshold = 0.1  # Ajusta este valor según tus mediciones
            if energy < energy_threshold:
                continue  # Saltar al siguiente ciclo de recepción

            # Detección de preámbulo usando Schmidl & Cox
            d_max, M = schmidl_cox_algorithm_vectorized(
                received_signal, L, threshold=preamble_threshold
            )
            if d_max is not None:
                if d_max == 0:  # Preámbulo coincide con la señal de inicio
                    print("El preámbulo coincide con la señal de inicio.")
                    try:
                        # Estimación del Desplazamiento de frecuencia portadora CFO
                        freq_offset_est = estimate_cfo(
                            received_signal,
                            d_max,
                            preamble_bits,
                            samples_per_symbol,
                            sample_rate,
                        )

                        # Corregir el CFO a partir de d_max
                        t = np.arange(len(received_signal[d_max:])) / sample_rate
                        received_signal[d_max:] *= np.exp(
                            -1j * 2 * np.pi * freq_offset_est * t
                        )

                        # Filtrado adaptado con filtro RRC
                        rrc_coef = rrc_filter(beta, samples_per_symbol, num_taps)
                        matched_filtered_signal = np.convolve(
                            received_signal, rrc_coef, mode="same"
                        )

                        # Sincronización en tiempo usando Mueller y Müller
                        timing_recovered_signal = mueller_muller_timing_recovery(
                            matched_filtered_signal[d_max:], samples_per_symbol
                        )

                        # Sincronización de frecuencia fina usando Costas Loop
                        frequency_corrected_signal = costas_loop(
                            timing_recovered_signal
                        )

                        # Demodulación
                        received_symbols = frequency_corrected_signal
                        received_bits = (np.real(received_symbols) > 0).astype(np.uint8)

                        # Extraer los bits del preámbulo recibidos
                        preamble_received_bits = received_bits[: len(preamble_bits)]
                        # Verificación del preámbulo
                        preamble_match = np.array_equal(
                            preamble_received_bits, preamble_bits
                        )
                        preamble_inverted_match = np.array_equal(
                            1 - preamble_received_bits, preamble_bits
                        )
                        if preamble_inverted_match:
                            print(
                                "Se detectó inversión de fase. Corrigiendo inversión de símbolos."
                            )
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
                        decoded_header_bits = channel_decode(
                            encoded_header_bits, repetition=repetition
                        )

                        # Convertir bits a enteros para obtener width y height
                        width_bits = decoded_header_bits[:16]
                        height_bits = decoded_header_bits[16:32]

                        # Convertir bits a bytes
                        width_bytes = np.packbits(width_bits)
                        height_bytes = np.packbits(height_bits)

                        # Convertir bytes a enteros de 16 bits
                        width = int(
                            np.frombuffer(width_bytes.tobytes(), dtype=">u2")[0]
                        )
                        height = int(
                            np.frombuffer(height_bytes.tobytes(), dtype=">u2")[0]
                        )

                        # Ahora, podemos calcular el número esperado de bits para la imagen
                        expected_image_bits = (
                            width * height * 3 * 8
                        )  # 3 canales (RGB), 8 bits por canal
                        expected_encoded_image_bits = expected_image_bits * repetition

                        # Verificar si tenemos suficientes bits recibidos
                        total_expected_bits = (
                            len(preamble_bits)
                            + header_length
                            + expected_encoded_image_bits
                        )

                        # Extraer y decodificar los bits de la imagen
                        encoded_image_bits = received_bits_rest[
                            header_length : header_length + expected_encoded_image_bits
                        ]
                        decoded_image_bits = channel_decode(
                            encoded_image_bits, repetition=repetition
                        )

                        # Convertir bits a bytes
                        image_bytes = np.packbits(decoded_image_bits)

                        # Convertir bytes a matriz de imagen
                        try:
                            image_array = np.reshape(image_bytes, (height, width, 3))
                            # Crear y mostrar la imagen
                            image = Image.fromarray(image_array, "RGB")
                            image.show()
                        except:
                            continue
                    except Exception as e:
                        print(f"Ocurrió un error durante el procesamiento: {e}")
                        continue  # Continuar con la siguiente iteración
                else:
                    continue
            else:
                print("No se detectó el preámbulo en la señal recibida.")
    except KeyboardInterrupt:
        print("Recepción interrumpida por el usuario.")
    finally:
        radio_rx.close()
