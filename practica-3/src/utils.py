import numpy as np

def rrc_filter(beta, sps, num_taps):
    """Genera un filtro de Coseno Elevado Raíz (RRC)."""
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / sps
    pi_t = np.pi * t
    four_beta_t = 4 * beta * t

    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.sin(pi_t * (1 - beta)) + 4 * beta * t * np.cos(pi_t * (1 + beta))
        denominator = pi_t * (1 - (four_beta_t) ** 2)
        h = numerator / denominator

    # Manejo de t = 0
    h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    # Manejo de t = ±1/(4β)
    t_special = np.abs(t) == (1 / (4 * beta))
    h[t_special] = (beta / np.sqrt(2)) * (
        ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) +
        ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    )

    # Normalizar el filtro
    h /= np.sqrt(np.sum(h**2))

    return h

def schmidl_cox_algorithm_vectorized(signal, L, threshold=0.8):
    """Algoritmo de Schmidl & Cox para detección de preámbulo."""
    # Calcular P(d)
    P = np.zeros(len(signal) - 2 * L + 1, dtype=complex)
    P = np.correlate(signal[L:], np.conj(signal[:-L]), mode='valid')

    # Calcular R(d)
    power = np.abs(signal)**2
    R = np.convolve(power[L:], np.ones(L), mode='valid')

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

def channel_encode(bits, repetition=3):
    """Codificación de canal por repetición."""
    return np.repeat(bits, repetition)

def channel_decode(bits, repetition=3):
    """Decodificación de canal por repetición."""
    bits = bits[:len(bits) - len(bits) % repetition]  # Ajustar longitud
    bits_reshaped = bits.reshape(-1, repetition)
    decoded_bits = (np.sum(bits_reshaped, axis=1) > (repetition / 2)).astype(np.uint8)
    return decoded_bits

def mueller_muller_timing_recovery(signal, sps):
    """Algoritmo de recuperación de temporización de Mueller y Müller."""
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

def costas_loop(signal, alpha=0.132, beta=0.00932):
    """Implementación del Costas Loop para sincronización de frecuencia fina."""
    N = len(signal)
    phase = 0
    freq = 0
    out = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        out[i] = signal[i] * np.exp(-1j * phase)
        error = np.real(out[i]) * np.imag(out[i])
        freq += beta * error
        phase += freq + alpha * error
    return out
