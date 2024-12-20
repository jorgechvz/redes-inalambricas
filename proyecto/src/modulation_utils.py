import numpy as np


# Funciones Comunes
def channel_encode(bits, repetition=3):
    return np.repeat(bits, repetition)


def channel_decode(bits, repetition=3):
    bits = bits[: len(bits) - len(bits) % repetition]  # Ajustar longitud
    bits_reshaped = bits.reshape(-1, repetition)
    decoded_bits = (np.sum(bits_reshaped, axis=1) > (repetition / 2)).astype(np.uint8)
    return decoded_bits


def schmidl_cox_algorithm_vectorized(signal, L, threshold=0.95):
    """Algoritmo de Schmidl & Cox para detección de preámbulo."""
    # Calcular P(d)
    P = np.zeros(len(signal) - 2 * L + 1, dtype=complex)
    P = np.correlate(signal[L:], np.conj(signal[:-L]), mode="valid")

    # Calcular R(d)
    power = np.abs(signal) ** 2
    R = np.convolve(power[L:], np.ones(L), mode="valid")

    # Calcular M(d)
    M = np.abs(P) / R
    M /= np.max(M)

    # Detectar picos
    peaks = np.where(M > threshold)[0]
    if len(peaks) > 0:
        d_max = peaks[0]
        return d_max, M
    else:
        return None, M


def mueller_muller_timing_recovery(signal, sps):
    """
    Recuperación de sincronización de tiempo utilizando el algoritmo de Mueller-Muller.
    """
    mu = 0.0  # Estimación inicial de fase
    out = []
    out_rail = []
    i_in = 0
    while i_in + int(mu) < len(signal):
        index = i_in + int(mu)
        if index >= len(signal):
            break
        sample = signal[int(index)]
        out.append(sample)
        # Para BPSK, solo necesitamos la parte real
        out_rail.append(np.sign(np.real(sample)))
        if len(out) >= 3:
            x = (out_rail[-1] - out_rail[-3]) * np.real(out[-2])
            mu += -x * 0.01  # Ajustar el factor según sea necesario
        else:
            mu += 0
        # Asegurarnos de que mu se mantenga en un rango razonable
        if mu > sps:
            mu -= sps
            i_in += sps
        else:
            i_in += sps
    return np.array(out)


def rrc_filter(beta, sps, num_taps):
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / sps
    pi_t = np.pi * t
    four_beta_t = 4 * beta * t

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = np.sin(pi_t * (1 - beta)) + 4 * beta * t * np.cos(pi_t * (1 + beta))
        denominator = pi_t * (1 - (four_beta_t) ** 2)
        h = numerator / denominator

    h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    t_special = np.abs(t) == (1 / (4 * beta))
    h[t_special] = (beta / np.sqrt(2)) * (
        ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)))
        + ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    )
    h /= np.sqrt(np.sum(h**2))
    return h


def bits_to_symbols(bits, modulation_scheme):
    if modulation_scheme == "BPSK":
        # BPSK: 1 bit por símbolo
        symbols = 2 * bits - 1
    elif modulation_scheme == "QPSK":
        # QPSK: 2 bits por símbolo
        bits_reshaped = bits.reshape(-1, 2)
        mapping = {
            (0, 0): (1 + 1j),
            (0, 1): (-1 + 1j),
            (1, 1): (-1 - 1j),
            (1, 0): (1 - 1j),
        }
        symbols = np.array([mapping[tuple(b)] for b in bits_reshaped]) / np.sqrt(2)
    elif modulation_scheme == "8QAM":
        # 8-QAM: 3 bits por símbolo
        symbols = bits_to_8qam_symbols(bits)
    elif modulation_scheme == "16QAM":
        # 16-QAM: 4 bits por símbolo
        symbols = bits_to_16qam_symbols(bits)
    else:
        raise ValueError("Modulación no soportada.")
    return symbols


def symbols_to_bits(symbols, modulation_scheme):
    if modulation_scheme == "BPSK":
        # BPSK: 1 bit por símbolo
        bits = (np.real(symbols) > 0).astype(
            np.uint8
        )  # Decisión basada en la parte real
    elif modulation_scheme == "QPSK":
        # Demodulación QPSK
        bits = qpsk_demodulate(symbols)
    elif modulation_scheme == "8QAM":
        # Demodulación 8-QAM
        bits = demodulate_8qam(symbols)
    elif modulation_scheme == "16QAM":
        # Demodulación 16-QAM
        bits = demodulate_16qam(symbols)
    else:
        raise ValueError("Modulación no soportada.")
    return bits


# Funciones para modulación QPSK


def costas_loop(signal, modulation_order=2, alpha=0.132, beta=0.00932):
    """
    Recuperación de sincronización de fase utilizando el algoritmo de Costas Loop.
    """
    N = len(signal)
    phase = 0
    freq = 0
    out = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        out[i] = signal[i] * np.exp(-1j * phase)
        if modulation_order == 2:  # BPSK
            error = np.sign(np.real(out[i])) * np.imag(out[i])
        elif modulation_order == 4:  # QPSK
            error = np.sign(np.real(out[i])) * np.imag(out[i]) - np.sign(
                np.imag(out[i])
            ) * np.real(out[i])
        else:
            # Para modulación de orden superior, se requiere un bucle de fase más complejo
            error = np.angle(out[i] ** modulation_order)
        freq += beta * error
        phase += freq + alpha * error
    return out


def qpsk_demodulate(symbols):
    bits = np.zeros((len(symbols), 2), dtype=np.uint8)
    bits[:, 0] = (np.real(symbols) < 0).astype(np.uint8)
    bits[:, 1] = (np.imag(symbols) < 0).astype(np.uint8)
    return bits.reshape(-1)


# Funcion para modulación 8-QAM


def bits_to_8qam_symbols(bits):
    bits = np.array(bits)
    # Asegurar que la longitud de bits sea múltiplo de 3
    num_bits = len(bits)
    num_symbols = num_bits // 3
    bits = bits[: num_symbols * 3]
    bits_reshaped = bits.reshape((num_symbols, 3))

    # Definir mapeo de símbolos
    symbol_mapping = {
        (0, 0, 0): (-1 - 1j) / np.sqrt(3),
        (0, 0, 1): (-1 + 1j) / np.sqrt(3),
        (0, 1, 0): (1 - 1j) / np.sqrt(3),
        (0, 1, 1): (1 + 1j) / np.sqrt(3),
        (1, 0, 0): (-2 + 0j) / np.sqrt(5),
        (1, 0, 1): (0 + 2j) / np.sqrt(5),
        (1, 1, 0): (2 + 0j) / np.sqrt(5),
        (1, 1, 1): (0 - 2j) / np.sqrt(5),
    }
    symbols = []
    for b in bits_reshaped:
        b_tuple = tuple(b)
        symbol = symbol_mapping[b_tuple]
        symbols.append(symbol)
    symbols = np.array(symbols)
    return symbols


def demodulate_8qam(symbols):
    # Definir constelación
    constellation = np.array(
        [
            (-1 - 1j) / np.sqrt(3),
            (-1 + 1j) / np.sqrt(3),
            (1 - 1j) / np.sqrt(3),
            (1 + 1j) / np.sqrt(3),
            (-2 + 0j) / np.sqrt(5),
            (0 + 2j) / np.sqrt(5),
            (2 + 0j) / np.sqrt(5),
            (0 - 2j) / np.sqrt(5),
        ]
    )
    # Mapear símbolos a bits
    bits_mapping = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    received_bits = []
    for symbol in symbols:
        distances = np.abs(symbol - constellation)
        index = np.argmin(distances)
        bits = bits_mapping[index]
        received_bits.extend(bits)
    return np.array(received_bits)


# Funcion para modulación 16-QAM


def bits_to_16qam_symbols(bits):
    bits = np.array(bits)
    # Asegurar que la longitud de bits sea múltiplo de 4
    num_bits = len(bits)
    num_symbols = num_bits // 4
    bits = bits[: num_symbols * 4]
    bits_reshaped = bits.reshape((num_symbols, 4))

    # Definir mapeo de símbolos para 16-QAM usando Gray code
    bits_to_levels = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}

    symbols = []
    for b in bits_reshaped:
        i_bits = tuple(b[:2])
        q_bits = tuple(b[2:])
        i_level = bits_to_levels[i_bits]
        q_level = bits_to_levels[q_bits]
        symbol = (i_level + 1j * q_level) / np.sqrt(
            10
        )  # Normalizar para potencia unitaria
        symbols.append(symbol)
    symbols = np.array(symbols)
    return symbols


def demodulate_16qam(symbols):
    # Definir constelación y mapeo de bits
    bits_list = [(0, 0), (0, 1), (1, 1), (1, 0)]  # -3  # -1  # 1  # 3
    bits_to_levels = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
    constellation = []
    bits_mapping = []
    for i_bits in bits_list:
        i_level = bits_to_levels[i_bits]
        for q_bits in bits_list:
            q_level = bits_to_levels[q_bits]
            symbol = (i_level + 1j * q_level) / np.sqrt(10)
            constellation.append(symbol)
            bits = i_bits + q_bits
            bits_mapping.append(bits)
    constellation = np.array(constellation)
    received_bits = []
    for symbol in symbols:
        distances = np.abs(symbol - constellation)
        index = np.argmin(distances)
        bits = bits_mapping[index]
        received_bits.extend(bits)
    return np.array(received_bits)
