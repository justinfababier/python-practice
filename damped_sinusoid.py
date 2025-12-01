import numpy as np
import matplotlib.pyplot as plt

def x(t, A, alpha, f, phi) -> float:
    """
    Damped sinusoid signal function.
    As time approaches infinite, the output approaches zero.

    Takes the following inputs:
        1) t (instantaneous time)
        2) A (initial amplitude)
        3) alpha (decay constant)
        4) f (frequency) of oscillation
        5) phi (phase shift)

    Returns:
        1) The amplitude of the signal as a float
    """
    return A * np.exp(-alpha * t) * np.sin((2 * np.pi * f * t) + phi)

def main():
    """
    Simple program that:
        1) Defines system duration, system sampling rate, and sampling times
        2) Defines and calculates the sampled signal 
        3) Plots the amplitude of the signal over time
    """
    duration = 15.0     # 15 seconds
    fs = 44100          # Sampling rate (44100 Hz, or 44.1 kHz)
    t_values = np.arange(0, duration, 1/fs) # Time values, linearlized

    # Sample signal
    samples = x(t_values, A=1, alpha=0.5, f=440, phi=0)

    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.title("Damped sinusoid signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.plot(t_values, samples, color='red')
    plt.show()

if __name__ == "__main__":
    main()