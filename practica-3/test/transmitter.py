import numpy as np
from PIL import Image

def rrc_filter(beta, sps, num_taps):
    """Genera un filtro de Coseno Elevado Raíz (RRC)."""
    t = np.arange(-num_taps//2, num_taps//2 + 1) / sps
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

def simulate_transmission(signal, snr_db):
    """Simula la transmisión sobre un canal AWGN."""
    # Calcular potencia de señal y ruido
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generar ruido blanco gaussiano
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )

    # Añadir ruido a la señal
    received_signal = signal + noise
    return received_signal

if __name__ == "__main__":
    # Parámetros de configuración
    samples_per_symbol = 8  # Muestras por símbolo
    beta = 0.2  # Factor de roll-off del filtro RRC
    num_taps = 101  # Número de taps del filtro RRC
    snr_db = 30  # Relación señal a ruido en dB

    # Desfase y CFO
    phase_offset = np.pi / 6  # Desfase de 30 grados
    frequency_offset = 0.01  # Desplazamiento de frecuencia normalizado

    # Semilla aleatoria para reproducibilidad
    np.random.seed(0)

    # Generar preámbulo
    half_preamble_bits = np.random.randint(0, 2, size=10)  # 10 bits
    preamble_bits = np.concatenate((half_preamble_bits, half_preamble_bits))
    preamble_symbols = 2 * preamble_bits - 1

    # Cargar imagen
    image = Image.open('imagen3.jpg')  
    image = image.convert('RGB')
    width, height = image.size
    image_array = np.array(image)
    image_flat = image_array.flatten()
    image_bits = np.unpackbits(image_flat)

    # Crear encabezado con el tamaño de la imagen
    width_bytes = np.array([width], dtype='>u2').view(np.uint8)
    height_bytes = np.array([height], dtype='>u2').view(np.uint8)
    width_bits = np.unpackbits(width_bytes)
    height_bits = np.unpackbits(height_bytes)
    header_bits = np.concatenate((width_bits, height_bits))

    # Concatenar preámbulo, encabezado y bits de imagen
    bits = np.concatenate((preamble_bits, header_bits, image_bits))

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

    # Aplicar desfase y CFO
    sample_rate = samples_per_symbol  # Tasa de muestreo normalizada
    t = np.arange(len(signal_filtered)) / sample_rate
    signal_complex = signal_filtered * np.exp(1j * (2 * np.pi * frequency_offset * t + phase_offset))

    # Simular transmisión sobre un canal AWGN
    received_signal = simulate_transmission(signal_complex, snr_db)

    # Guardar la señal recibida para el receptor
    np.save('received_signal.npy', received_signal)

    print("Simulación de transmisión completa. Señal guardada para el procesamiento del receptor.")
