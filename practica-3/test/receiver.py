# Receiver.py

import numpy as np
from PIL import Image  # Import PIL for image handling

def schmidl_cox_algorithm_vectorized(signal, L, threshold=0.8):
    N = len(signal)
    # Cálculo de P(d)
    P = np.array([
        np.sum(signal[d:d+L] * np.conj(signal[d+L:d+2*L]))
        for d in range(N - 2*L)
    ])
    # Cálculo de R(d)
    R = np.array([
        np.sum(np.abs(signal[d+L:d+2*L])**2)
        for d in range(N - 2*L)
    ])
    M = np.abs(P) / R
    M = M / np.max(M)
    peaks = np.where(M > threshold)[0]
    if len(peaks) > 0:
        d_max = peaks[0]
        return d_max, M
    else:
        return None, M


def rrc_filter(beta, sps, num_taps):
    """Generates a Root Raised Cosine (RRC) filter."""
    # Same as in transmitter
    t = np.arange(-num_taps//2, num_taps//2 + 1) / sps
    pi_t = np.pi * t
    four_beta_t = 4 * beta * t

    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.sin(pi_t * (1 - beta)) + 4 * beta * t * np.cos(pi_t * (1 + beta))
        denominator = pi_t * (1 - (four_beta_t) ** 2)
        h = numerator / denominator

    # Handle t = 0 case
    h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    # Handle t = ±1/(4β)
    t_special = np.abs(t) == (1 / (4 * beta))
    h[t_special] = (beta / np.sqrt(2)) * (
        ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) +
        ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    )

    # Normalize the filter
    h /= np.sqrt(np.sum(h**2))

    return h

if __name__ == "__main__":
    samples_per_symbol = 8
    beta = 0.2
    num_taps = 101

    np.random.seed(0)

    half_preamble_bits = np.random.randint(0, 2, size=10)
    preamble_bits = np.concatenate((half_preamble_bits, half_preamble_bits))
    preamble_symbols = 2 * preamble_bits - 1

    received_signal = np.load('received_signal.npy')

    L = len(half_preamble_bits) * samples_per_symbol
    print(f"Longitud de la mitad del preámbulo en muestras: {L}")
    d_max, M = schmidl_cox_algorithm_vectorized(received_signal, L)

    if d_max is not None:
        print(f"Preámbulo detectado en el índice: {d_max}")

        # Estimación del desplazamiento de frecuencia
        A = np.vdot(received_signal[d_max:d_max+L], received_signal[d_max+L:d_max+2*L])
        freq_offset_est = np.angle(A) / (2 * np.pi * L)
        print(f"Desplazamiento de frecuencia estimado (CFO): {freq_offset_est}")

        # Corregir el desplazamiento de frecuencia
        sample_rate = samples_per_symbol
        t = np.arange(len(received_signal)) / sample_rate
        received_signal *= np.exp(-1j * 2 * np.pi * freq_offset_est * t)

        # Filtrado adaptado con filtro RRC
        rrc_coef = rrc_filter(beta, samples_per_symbol, num_taps)
        matched_filtered_signal = np.convolve(received_signal, rrc_coef, mode='same')

        # Introducir error de muestreo
        timing_offset = 0  # Desplazamiento de tiempo en muestras

        # Ajustar symbol_indices para comenzar desde d_max + timing_offset
        symbol_indices = d_max + np.arange(0, len(matched_filtered_signal) - d_max, samples_per_symbol)

        # Interpolación para muestreo
        from scipy.interpolate import interp1d
        interp_real = interp1d(
            np.arange(len(matched_filtered_signal)),
            np.real(matched_filtered_signal),
            kind='linear',
            fill_value="extrapolate"
        )
        interp_imag = interp1d(
            np.arange(len(matched_filtered_signal)),
            np.imag(matched_filtered_signal),
            kind='linear',
            fill_value="extrapolate"
        )
        sampled_real = interp_real(symbol_indices)
        sampled_imag = interp_imag(symbol_indices)
        sampled_symbols = sampled_real + 1j * sampled_imag

        # Extraer los símbolos recibidos correspondientes al preámbulo
        preamble_received = sampled_symbols[:len(preamble_symbols)]

        # Estimación y corrección del desfase
        phase_estimate = np.angle(np.mean(preamble_received * preamble_symbols.conj()))
        print(f"Desfase estimado: {phase_estimate}")

        # Corregir el desfase
        sampled_symbols *= np.exp(-1j * phase_estimate)

        # Decodificar símbolos a bits
        received_bits = (np.real(sampled_symbols) > 0).astype(np.uint8)

        # Extraer bits de encabezado (32 bits)
        header_bits = received_bits[len(preamble_bits):len(preamble_bits)+32]

        # Convertir bits de encabezado a ancho y alto
        width_bits = header_bits[:16]
        height_bits = header_bits[16:32]

        width_bytes = np.packbits(width_bits)
        height_bytes = np.packbits(height_bits)

        # Convertir bytes a enteros
        width = int(np.frombuffer(width_bytes.tobytes(), dtype='>u2')[0])
        height = int(np.frombuffer(height_bytes.tobytes(), dtype='>u2')[0])

        print(f"Ancho de imagen: {width}, Alto de imagen: {height}")

        # Calcular el número de bits para la imagen
        num_image_bits = width * height * 3 * 8  # 3 colores (RGB), 8 bits por color

        # Extraer bits de imagen
        total_bits_needed = len(preamble_bits) + 32 + num_image_bits
        if len(received_bits) < total_bits_needed:
            print("No se recibieron suficientes bits para reconstruir la imagen.")
        else:
            image_bits = received_bits[len(preamble_bits)+32:total_bits_needed]

            # Convertir bits a bytes
            image_bytes = np.packbits(image_bits)

            # Convertir bytes a matriz de imagen
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)

            # Verificar tamaño esperado
            expected_size = width * height * 3
            if image_array.size != expected_size:
                print(f"Tamaño esperado de la imagen: {expected_size}, pero se obtuvo: {image_array.size}")
                print("No se puede reestructurar la matriz debido a una discrepancia de tamaño.")
            else:
                image_array = image_array.reshape((height, width, 3))

                # Crear y mostrar la imagen
                image = Image.fromarray(image_array, 'RGB')
                image.show()

    else:
        print("No se detectó el preámbulo en la señal recibida.")