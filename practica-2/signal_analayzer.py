import numpy as np
import matplotlib.pyplot as plt

def shift_frequency(signal, freq_offset):
    n = np.arange(len(signal))
    return signal * np.exp(2j * np.pi * freq_offset * n)

def schmidl_cox_algorithm(signal, L, threshold=0.8):
    N = len(signal)
    P = np.zeros(N - 2 * L, dtype=complex)
    R = np.zeros(N - 2 * L)
    M = np.zeros(N - 2 * L)

    for d in range(N - 2 * L):
        P[d] = np.sum(signal[d:d+L] * np.conj(signal[d+L:d+2*L]))
        R[d] = np.sum(np.abs(signal[d:d+2*L])**2)
        if R[d] != 0:
            M[d] = np.abs(P[d])**2 / R[d]**2
        else:
            M[d] = 0

    max_M = np.max(M)
    if max_M > 0:
        M = M / max_M
    else:
        M = np.zeros_like(M)

    peaks = np.where(M > threshold)[0]
    if len(peaks) > 0:
        d_max = peaks[0]
        return d_max, M
    else:
        return None, M

def detect_preamble_with_freq_offset(signal, short_preamble_len, freq_offsets):
    best_d_max = None
    best_freq_offset = None
    best_M = None
    
    for freq_offset in freq_offsets:
        shifted_signal = shift_frequency(signal, freq_offset)
        d_max, M = schmidl_cox_algorithm(shifted_signal, short_preamble_len)
        
        if d_max is not None:
            best_d_max = d_max
            best_freq_offset = freq_offset
            best_M = M
            break
    
    return best_d_max, best_freq_offset, best_M

def rrc_filter(beta, sps, num_taps):
    t = np.arange(-num_taps//2, num_taps//2 + 1) / sps
    pi_t = np.pi * t
    four_beta_t = 4 * beta * t

    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.sin(pi_t * (1 - beta)) + 4 * beta * t * np.cos(pi_t * (1 + beta))
        denominator = pi_t * (1 - (four_beta_t) ** 2)
        h = numerator / denominator

    h[np.isnan(h)] = 1.0 - beta + (4 * beta / np.pi)
    t_special = np.abs(t) == (1 / (4 * beta))
    h[t_special] = (beta / np.sqrt(2)) * (
        ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) +
        ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    )

    h /= np.sqrt(np.sum(h**2))

    return h

class SignalAnalyzer:
    def __init__(self, sample_rate, preamble_len, samples_per_symbol, beta, num_taps):
        self.sample_rate = sample_rate
        self.preamble_len = preamble_len
        self.samples_per_symbol = samples_per_symbol
        self.beta = beta
        self.num_taps = num_taps

    def load_samples_from_file(self, filename):
        samples = np.load(filename)
        return samples

    def process_preamble(self, samples):
        rrc_coef = rrc_filter(self.beta, self.samples_per_symbol, self.num_taps)
        samples_filtered = np.convolve(samples, rrc_coef, mode='same')
        L = (self.preamble_len * self.samples_per_symbol) // 2

        d_max, M = schmidl_cox_algorithm(samples_filtered, L)
        return d_max, M

    def calculate_spectrum(self, samples):
        spectrum = np.fft.fftshift(np.fft.fft(samples))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1 / self.sample_rate))
        magnitude = 20 * np.log10(np.abs(spectrum) + 1e-12)
        return freqs, magnitude

class SpectrumPreamblePlotter:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.init_plot()
        self.preamble_detected = False

    def init_plot(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Para la señal en el dominio del tiempo
        self.line1_time, = self.ax1.plot([], [], lw=2)
        self.ax1.set_title("Señal en el Dominio del Tiempo")
        self.ax1.set_xlabel("Muestras")
        self.ax1.set_ylabel("Amplitud")
        self.ax1.grid(True)

        # Para el espectro de la señal
        self.line2_spectrum, = self.ax2.plot([], [], lw=2)
        self.ax2.set_title("Espectro de la Señal")
        self.ax2.set_xlabel("Frecuencia (MHz)")
        self.ax2.set_ylabel("Magnitud (dB)")
        self.ax2.grid(True)

        # Para la detección del preámbulo
        self.line3_preamble, = self.ax3.plot([], [], lw=2, label="M(d)")
        self.ax3.set_title("Detección de Preámbulo usando Schmidl-Cox")
        self.ax3.set_xlabel("Desplazamiento (d)")
        self.ax3.set_ylabel("M(d)")
        self.ax3.grid(True)
        self.ax3.legend()
        plt.tight_layout()

    def update_time_plot(self, samples):
        self.line1_time.set_data(np.arange(len(samples)), np.real(samples))
        self.ax1.set_xlim(0, len(samples))
        self.ax1.set_ylim(np.min(np.real(samples)), np.max(np.real(samples)))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_spectrum_plot(self, freqs, magnitude):
        self.line2_spectrum.set_data(freqs / 1e6, magnitude)
        self.ax2.set_xlim(freqs[0] / 1e6, freqs[-1] / 1e6)
        self.ax2.set_ylim(np.min(magnitude) - 10, np.max(magnitude) + 10)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_preamble_plot(self, M, d_max):
        self.line3_preamble.set_data(range(len(M)), M)
        self.ax3.set_xlim(0, len(M))
        self.ax3.set_ylim(0, 1.1)
        if d_max is not None:
            self.ax3.axvline(d_max, color='r', linestyle='--', label="Preámbulo Detectado")
            self.ax3.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def start_processing(self, samples):
        # Mostrar la señal en el dominio del tiempo
        self.update_time_plot(samples)

        # Calcular y mostrar el espectro
        freqs, magnitude = self.analyzer.calculate_spectrum(samples)
        self.update_spectrum_plot(freqs, magnitude)

        # Detectar el preámbulo
        freq_offsets = np.linspace(-0.001, 0.001, 10)
        short_preamble_length = self.analyzer.preamble_len // 2
        d_max, best_freq_offset, M = detect_preamble_with_freq_offset(
            samples, short_preamble_length, freq_offsets)

        # Mostrar resultados
        self.update_preamble_plot(M, d_max)
        if d_max is not None:
            print(f"Preámbulo detectado en d = {d_max} con desplazamiento de frecuencia = {best_freq_offset}")
        else:
            print("No se detectó el preámbulo.")

        # Mantener la visualización activa
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    sample_rate = 5e6
    preamble_len = 200
    samples_per_symbol = 8
    beta = 0.2
    num_taps = 151

    analyzer = SignalAnalyzer(sample_rate, preamble_len, samples_per_symbol, beta, num_taps)
    samples = analyzer.load_samples_from_file('received_signal.npy')

    plotter = SpectrumPreamblePlotter(analyzer)
    plotter.start_processing(samples)
