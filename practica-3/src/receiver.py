import numpy as np
from PIL import Image
from bladerf import _bladerf
from utils import rrc_filter, schmidl_cox_algorithm_vectorized, mueller_muller_timing_recovery, costas_loop, channel_decode
import matplotlib.pyplot as plt

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

    def receive_samples(self, num_samples):
        """Recibe muestras del BladeRF y las devuelve como un array de números complejos."""
        # Crear un buffer para recibir los datos crudos (bytes)
        bytes_per_sample = 4  # 2 bytes para I y 2 bytes para Q
        buf = bytearray(num_samples * bytes_per_sample)

        # Configurar el stream síncrono para la recepción
        self.sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X1,
                             fmt=_bladerf.Format.SC16_Q11,
                             num_buffers=16,
                             buffer_size=8192,
                             num_transfers=8,
                             stream_timeout=3500)

        # Iniciar la recepción
        print("Iniciando recepción...")
        self.rx_ch.enable = True  # Habilitar el canal de recepción

        self.sdr.sync_rx(buf, num_samples)  # Recibir los datos del BladeRF

        # Finalizar la recepción
        print("Deteniendo recepción")
        self.rx_ch.enable = False  # Deshabilitar el canal de recepción
        self.sdr.close()

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

if __name__ == "__main__":
    # Parámetros de configuración
    sample_rate = 5e6  # Tasa de muestreo
    center_freq = 940e6  # Frecuencia central
    gain = 20  # Ganancia
    samples_per_symbol = 8  # Número de muestras por símbolo
    beta = 0.2  # Factor de rodadura del filtro RRC
    num_taps = 101  # Número de coeficientes del filtro RRC
    repetition = 3  # Repetición en codificación de canal

    # Instanciar el receptor
    radio_rx = BladeRFReceiver(sample_rate, center_freq, gain)

    # Número de muestras a recibir
    num_samples = 500000  

    # Recibir muestras
    received_signal = radio_rx.receive_samples(num_samples)

    # Guardar la señal recibida para análisis
    np.save('../datareceived_signal_from_bladerf.npy', received_signal)

    # Cargar bits del preámbulo desde el archivo
    preamble_bits = np.load('/practica-3/data/preamble_bits.npy')
    preamble_symbols = 2 * preamble_bits - 1
    L = len(preamble_symbols) * samples_per_symbol

    # Detección de preámbulo usando Schmidl & Cox
    d_max, M = schmidl_cox_algorithm_vectorized(received_signal, L)
    if d_max is not None:
        print(f"Preámbulo detectado en el índice: {d_max}")
        # Estimación del Desplazamiento de frecuencia portadora CFO
        A = np.sum(received_signal[d_max:d_max+L] * np.conj(received_signal[d_max+L:d_max+2*L]))
        freq_offset_est = (sample_rate / (2 * np.pi * L)) * np.angle(A)
        print(f"Desplazamiento de frecuencia estimado (CFO): {freq_offset_est}")
        # Corregir el Desplazamiento de frecuencia portadora CFO
        t = np.arange(len(received_signal)) / sample_rate
        received_signal *= np.exp(-1j * 2 * np.pi * freq_offset_est * t)

        # Filtrado adaptado con filtro RRC
        rrc_coef = rrc_filter(beta, samples_per_symbol, num_taps)
        matched_filtered_signal = np.convolve(received_signal, rrc_coef, mode='same')

        # Sincronización en tiempo usando Mueller y Müller
        timing_recovered_signal = mueller_muller_timing_recovery(matched_filtered_signal[d_max:], samples_per_symbol)

        # Sincronización de frecuencia fina usando Costas Loop
        frequency_corrected_signal = costas_loop(timing_recovered_signal)

        # Demodulación 
        received_symbols = frequency_corrected_signal
        received_bits = (np.real(received_symbols) > 0).astype(np.uint8)

        # Eliminar posibles bits adicionales
        transmitted_bits = np.load('/practica-3/data/transmitted_bits.npy')
        received_bits = received_bits[:len(transmitted_bits)]

        # Extraer los bits del preámbulo recibidos
        preamble_received_bits = received_bits[:len(preamble_bits)]

        # Verificación del preámbulo
        preamble_match = np.array_equal(preamble_received_bits, preamble_bits)
        print(f'Preamble bits match: {preamble_match}')

        if not preamble_match:
            print("El preámbulo no coincide. No se puede continuar.")
        else:
            # Ahora aplicar la decodificación de canal al resto de los bits
            received_bits_rest = received_bits[len(preamble_bits):]
            decoded_bits_rest = channel_decode(received_bits_rest, repetition=repetition)

            # Concatenar los bits decodificados
            decoded_bits = np.concatenate((preamble_bits, decoded_bits_rest))

            # Extraer y decodificar el encabezado
            header_bits = decoded_bits[len(preamble_bits):len(preamble_bits) + 32]
            width_bits = header_bits[:16]
            height_bits = header_bits[16:32]

            # Convertir bits a bytes
            width_bytes = np.packbits(width_bits)
            height_bytes = np.packbits(height_bits)

            # Convertir bytes a enteros de 16 bits
            width = int(np.frombuffer(width_bytes.tobytes(), dtype='>u2')[0])
            height = int(np.frombuffer(height_bytes.tobytes(), dtype='>u2')[0])

            print(f"Ancho de imagen: {width}, Alto de imagen: {height}")

            # Extraer bits de imagen
            image_bits = decoded_bits[len(preamble_bits) + 32:]
            # Ajustar longitud
            expected_bits = width * height * 3 * 8  # 3 canales (RGB), 8 bits por canal
            image_bits = image_bits[:expected_bits]

            # Convertir bits a bytes
            image_bytes = np.packbits(image_bits)
            # Convertir bytes a matriz de imagen
            try:
                image_array = np.reshape(image_bytes, (height, width, 3))
                # Crear y mostrar la imagen
                image = Image.fromarray(image_array, 'RGB')
                image.show()
                print("Imagen reconstruida correctamente.")
            except ValueError as e:
                print(f"Error al reconstruir la imagen: {e}")
    else:
        print("No se detectó el preámbulo en la señal recibida.")

    # Visualización de la constelación final
    if 'frequency_corrected_signal' in locals():
        plt.figure(figsize=(6,6))
        plt.scatter(np.real(frequency_corrected_signal), np.imag(frequency_corrected_signal), color='blue', alpha=0.5)
        plt.title('Constelación BPSK después de Todas las Sincronizaciones')
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginaria')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
