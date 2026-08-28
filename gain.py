# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scikit-rf",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gain - How to Read an RF Amplifier Datasheet
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import skrf as rf

    plt.style.use("seaborn-v0_8")
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "figure.titlesize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
        },
    )
    return np, plt, rf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decibel (dB) Scale

    Voltage gain:

    $$
        \large G = 20 \log_{10}{\left(\frac{V_{out}}{V_{in}}\right)}
    $$

    Power gain:

    $$
        \large G = 10 \log_{10}{\left(\frac{P_{out}}{P_{in}}\right)}
    $$
    """)
    return


@app.cell
def _(mo, np, plt):
    lin = np.linspace(1 / 1000, 100, 1000)
    power_db = 10 * np.log10(lin)
    voltage_db = 20 * np.log10(lin)
    _fig, _ax = plt.subplots()
    ax2 = _ax.twinx()
    (p1,) = _ax.plot(lin, lin, c="b", label="$x$")
    (p2,) = ax2.plot(lin, power_db, c="r", label="Power | $10 \\log_{10}(x)$")
    (p3,) = ax2.plot(
        lin, voltage_db, c="y", label="Voltage | $20 \\log_{10}(x)$"
    )
    lines = [p1, p2, p3]
    labels = [l.get_label() for l in lines]
    legend = _ax.legend(
        lines,
        labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.8),
        frameon=True,
        fancybox=True,
        framealpha=1,
        edgecolor="black",
    )
    _ax.set_xlabel("x")
    _ax.set_ylabel("Linear (V/V or W/W)")
    ax2.set_ylabel("dB")
    _ax.set_title("Linear / Log (dB) Scale Comparison")
    mo.vstack(
        [_fig, mo.Html("<div style='height: 24px'></div>")],
        gap=0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ADL8142
    K-band low-noise amplifier (LNA)

    [Datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/adl8142.pdf)

    You can download the s-parameters from the [product page](https://www.analog.com/en/products/adl8142.html#tools-header).
    """)
    return


@app.cell
def _(mo, rf):
    import pathlib
    import urllib.request

    # Resolves to a local path when run locally, and to a URL when the
    # notebook is exported to WASM and served over HTTP.
    s2p_source = str(
        mo.notebook_location() / "public" / "ADL8142ACPZN_25degC.s2p"
    )
    if s2p_source.startswith("http"):
        s2p_path = pathlib.Path("ADL8142ACPZN_25degC.s2p")
        s2p_path.write_bytes(urllib.request.urlopen(s2p_source).read())
    else:
        s2p_path = pathlib.Path(s2p_source)

    amp = rf.Network(str(s2p_path))
    return (amp,)


@app.cell
def _(amp, mo, plt):
    bw_mask = (amp.f > 23e9) & (amp.f < 31e9)
    _fig, _ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    _fig.suptitle("ADL8142 S-Parameters", y=0.85)
    _ax[0][0].set_ylim(-40, 5)
    _ax[0][0].set_title("Input Return Loss | $S_{11}$")
    _ax[0][0].plot(amp.f[bw_mask] / 1000000000.0, amp.s_db[bw_mask][:, 0, 0])
    _ax[0][1].set_ylim(20, 35)
    _ax[0][1].set_title("Gain | $S_{21}$")
    _ax[0][1].plot(amp.f[bw_mask] / 1000000000.0, amp.s_db[bw_mask][:, 1, 0])
    _ax[1][0].set_ylim(-80, 5)
    _ax[1][0].set_title("Reverse Gain | $S_{12}$")
    _ax[1][0].plot(amp.f[bw_mask] / 1000000000.0, amp.s_db[bw_mask][:, 0, 1])
    _ax[1][1].set_ylim(-40, 5)
    _ax[1][1].set_title("Output Return Loss | $S_{22}$")
    _ax[1][1].plot(amp.f[bw_mask] / 1000000000.0, amp.s_db[bw_mask][:, 1, 1])
    _fig.tight_layout(rect=[0, 0, 1, 0.93], pad=4, h_pad=5, w_pad=3)
    mo.vstack(
        [_fig, mo.Html("<div style='height: 24px'></div>")],
        gap=0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Amplifying a linear frequency-modulated (LFM) signal
    """)
    return


@app.cell
def _(amp, mo, np, plt):
    start_frequency_hz = 23e9
    stop_frequency_hz = 31e9
    sample_rate_hz = 80e9
    pulse_duration_s = 20e-9

    sample_count = int(sample_rate_hz * pulse_duration_s)
    time_s = np.arange(sample_count) / sample_rate_hz
    chirp_rate_hz_s = (
        stop_frequency_hz - start_frequency_hz
    ) / pulse_duration_s
    lfm_waveform = np.cos(
        2
        * np.pi
        * (start_frequency_hz * time_s + chirp_rate_hz_s * time_s**2 / 2)
    )

    frequency_hz = np.fft.rfftfreq(sample_count, 1 / sample_rate_hz)
    lfm_spectrum = np.fft.rfft(lfm_waveform)
    adl8142_gain_db = np.interp(frequency_hz, amp.f, amp.s_db[:, 1, 0])
    adl8142_voltage_gain = 10 ** (adl8142_gain_db / 20)
    amplified_spectrum = lfm_spectrum * adl8142_voltage_gain
    amplified_waveform = np.fft.irfft(amplified_spectrum, n=sample_count)

    spectrum_reference = np.max(np.abs(lfm_spectrum))
    lfm_spectrum_db = 20 * np.log10(
        np.maximum(np.abs(lfm_spectrum), np.finfo(float).tiny)
        / spectrum_reference
    )
    amplified_spectrum_db = 20 * np.log10(
        np.maximum(np.abs(amplified_spectrum), np.finfo(float).tiny)
        / spectrum_reference
    )
    lfm_band_mask = (frequency_hz > 20e9) & (frequency_hz < 34e9)

    _fig, _ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    _fig.suptitle("LFM Through ADL8142", y=0.85)
    _ax[0][0].set_title("LFM Time Series")
    _ax[0][0].plot(time_s / 1e-9, lfm_waveform)
    _ax[0][0].set_xlabel("Time (ns)")
    _ax[0][0].set_ylabel("Amplitude")
    _ax[0][1].set_title("LFM Spectrum")
    _ax[0][1].plot(
        frequency_hz[lfm_band_mask] / 1e9,
        lfm_spectrum_db[lfm_band_mask],
    )
    _ax[0][1].set_xlabel("Frequency (GHz)")
    _ax[0][1].set_ylabel("Magnitude (dB)")
    _ax[1][0].set_title("ADL8142 Output Time Series")
    _ax[1][0].plot(time_s / 1e-9, amplified_waveform)
    _ax[1][0].set_xlabel("Time (ns)")
    _ax[1][0].set_ylabel("Amplitude")
    _ax[1][1].set_title("ADL8142 Output Spectrum")
    _ax[1][1].plot(
        frequency_hz[lfm_band_mask] / 1e9,
        amplified_spectrum_db[lfm_band_mask],
    )
    _ax[1][1].set_xlabel("Frequency (GHz)")
    _ax[1][1].set_ylabel("Magnitude (dB)")
    _fig.tight_layout(rect=[0, 0, 1, 0.93], pad=4, h_pad=5, w_pad=3)
    mo.vstack(
        [_fig, mo.Html("<div style='height: 24px'></div>")],
        gap=0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Amplifying a multi-frequency signal
    """)
    return


@app.cell
def _(mo, np, plt):
    illustrative_sample_rate_hz = 200e6
    illustrative_duration_s = 1e-6
    illustrative_sample_count = int(
        illustrative_sample_rate_hz * illustrative_duration_s
    )
    illustrative_time_s = (
        np.arange(illustrative_sample_count) / illustrative_sample_rate_hz
    )
    component_frequencies_hz = np.array([5e6, 15e6, 30e6])
    input_signal = np.cos(
        2 * np.pi * component_frequencies_hz[:, None] * illustrative_time_s
    ).sum(axis=0)
    illustrative_fft_frequency_hz = np.fft.rfftfreq(
        illustrative_sample_count,
        1 / illustrative_sample_rate_hz,
    )
    input_spectrum = np.fft.rfft(input_signal)
    input_spectrum_amplitude = (
        2 * np.abs(input_spectrum) / illustrative_sample_count
    )
    illustrative_frequency_mask = illustrative_fft_frequency_hz <= 40e6

    _fig, _ax = plt.subplots(nrows=2, figsize=(12, 8))
    _ax[0].set_title("Input Time Series")
    _ax[0].plot(illustrative_time_s / 1e-6, input_signal, c="C0")
    _ax[0].set_xlabel(r"Time ($\mu$s)")
    _ax[0].set_ylabel("Amplitude")
    _ax[1].set_title("Input Frequency Spectrum")
    _ax[1].plot(
        illustrative_fft_frequency_hz[illustrative_frequency_mask] / 1e6,
        input_spectrum_amplitude[illustrative_frequency_mask],
        c="C0",
    )
    _ax[1].set_xlabel("Frequency (MHz)")
    _ax[1].set_ylabel("Amplitude")
    _fig.tight_layout(pad=4, h_pad=5)
    mo.vstack(
        [_fig, mo.Html("<div style='height: 24px'></div>")],
        gap=0,
    )
    return (
        component_frequencies_hz,
        illustrative_fft_frequency_hz,
        illustrative_frequency_mask,
        illustrative_sample_count,
        illustrative_time_s,
        input_signal,
        input_spectrum,
        input_spectrum_amplitude,
    )


@app.cell
def _(component_frequencies_hz, illustrative_fft_frequency_hz, mo, np, plt):
    illustrative_amp_frequency_hz = (
        np.array([0, 5, 10, 15, 20, 25, 30, 40, 100]) * 1e6
    )
    illustrative_amp_gain_db = np.array([0, 12, 8, 3, 0, -4, -9, -15, -20])
    interpolated_amp_gain_db = np.interp(
        illustrative_fft_frequency_hz,
        illustrative_amp_frequency_hz,
        illustrative_amp_gain_db,
    )
    illustrative_voltage_gain = 10 ** (interpolated_amp_gain_db / 20)

    _fig, _ax = plt.subplots(figsize=(12, 6))
    _ax.set_title("Illustrative Amplifier Gain")
    _ax.plot(
        illustrative_amp_frequency_hz / 1e6,
        illustrative_amp_gain_db,
    )
    _ax.axhline(0, c="k", linewidth=1)
    _ax.scatter(
        component_frequencies_hz / 1e6,
        np.interp(
            component_frequencies_hz,
            illustrative_amp_frequency_hz,
            illustrative_amp_gain_db,
        ),
        c="r",
    )
    _ax.set_xlim(0, 40)
    _ax.set_xlabel("Frequency (MHz)")
    _ax.set_ylabel("Gain (dB)")
    _fig.tight_layout(pad=4)
    mo.vstack(
        [_fig, mo.Html("<div style='height: 24px'></div>")],
        gap=0,
    )
    return (
        illustrative_amp_frequency_hz,
        illustrative_amp_gain_db,
        illustrative_voltage_gain,
    )


@app.cell
def _(
    illustrative_fft_frequency_hz,
    illustrative_frequency_mask,
    illustrative_sample_count,
    illustrative_time_s,
    illustrative_voltage_gain,
    input_signal,
    input_spectrum,
    input_spectrum_amplitude,
    mo,
    np,
    plt,
):
    output_spectrum = input_spectrum * illustrative_voltage_gain
    output_signal = np.fft.irfft(output_spectrum, n=illustrative_sample_count)
    output_spectrum_amplitude = (
        2 * np.abs(output_spectrum) / illustrative_sample_count
    )
    time_amplitude_limit = 1.1 * max(
        np.max(np.abs(input_signal)),
        np.max(np.abs(output_signal)),
    )
    spectrum_amplitude_limit = 1.1 * max(
        np.max(input_spectrum_amplitude),
        np.max(output_spectrum_amplitude),
    )

    _fig, _ax = plt.subplots(nrows=2, figsize=(12, 8))
    _ax[0].set_title("Output Time Series")
    _ax[0].plot(illustrative_time_s / 1e-6, output_signal, c="C1")
    _ax[0].set_xlabel(r"Time ($\mu$s)")
    _ax[0].set_ylabel("Amplitude")
    _ax[1].set_title("Output Frequency Spectrum")
    _ax[1].plot(
        illustrative_fft_frequency_hz[illustrative_frequency_mask] / 1e6,
        output_spectrum_amplitude[illustrative_frequency_mask],
        c="C1",
    )
    _ax[1].set_xlabel("Frequency (MHz)")
    _ax[1].set_ylabel("Amplitude")
    _fig.tight_layout(pad=4, h_pad=5)
    mo.vstack(
        [_fig, mo.Html("<div style='height: 24px'></div>")],
        gap=0,
    )
    return (
        output_signal,
        output_spectrum_amplitude,
        spectrum_amplitude_limit,
        time_amplitude_limit,
    )


@app.cell
def _(
    component_frequencies_hz,
    illustrative_amp_frequency_hz,
    illustrative_amp_gain_db,
    illustrative_fft_frequency_hz,
    illustrative_frequency_mask,
    illustrative_time_s,
    input_signal,
    input_spectrum_amplitude,
    mo,
    np,
    output_signal,
    output_spectrum_amplitude,
    plt,
    spectrum_amplitude_limit,
    time_amplitude_limit,
):
    _fig, _ax = plt.subplot_mosaic(
        [
            ["input_time", "input_frequency"],
            ["amplifier_gain", "amplifier_gain"],
            ["output_time", "output_frequency"],
        ],
        figsize=(12, 12),
        gridspec_kw={"height_ratios": [1, 0.8, 1]},
    )
    _fig.suptitle("Three-Tone Signal Through an Illustrative Amplifier")
    _ax["input_time"].set_title("Input Time Series")
    _ax["input_time"].plot(illustrative_time_s / 1e-6, input_signal, c="C0")
    _ax["input_time"].set_ylim(-time_amplitude_limit, time_amplitude_limit)
    _ax["input_time"].set_xlabel(r"Time ($\mu$s)")
    _ax["input_time"].set_ylabel("Amplitude")
    _ax["input_frequency"].set_title("Input Frequency Spectrum")
    _ax["input_frequency"].plot(
        illustrative_fft_frequency_hz[illustrative_frequency_mask] / 1e6,
        input_spectrum_amplitude[illustrative_frequency_mask],
        c="C0",
    )
    _ax["input_frequency"].set_ylim(0, spectrum_amplitude_limit)
    _ax["input_frequency"].set_xlabel("Frequency (MHz)")
    _ax["input_frequency"].set_ylabel("Amplitude")
    _ax["amplifier_gain"].set_title("Illustrative Amplifier Gain")
    _ax["amplifier_gain"].plot(
        illustrative_amp_frequency_hz / 1e6,
        illustrative_amp_gain_db,
    )
    _ax["amplifier_gain"].axhline(0, c="k", linewidth=1)
    _ax["amplifier_gain"].scatter(
        component_frequencies_hz / 1e6,
        np.interp(
            component_frequencies_hz,
            illustrative_amp_frequency_hz,
            illustrative_amp_gain_db,
        ),
        c="r",
    )
    _ax["amplifier_gain"].set_xlim(0, 40)
    _ax["amplifier_gain"].set_xlabel("Frequency (MHz)")
    _ax["amplifier_gain"].set_ylabel("Gain (dB)")
    _ax["output_time"].set_title("Output Time Series")
    _ax["output_time"].plot(illustrative_time_s / 1e-6, output_signal, c="C1")
    _ax["output_time"].set_ylim(-time_amplitude_limit, time_amplitude_limit)
    _ax["output_time"].set_xlabel(r"Time ($\mu$s)")
    _ax["output_time"].set_ylabel("Amplitude")
    _ax["output_frequency"].set_title("Output Frequency Spectrum")
    _ax["output_frequency"].plot(
        illustrative_fft_frequency_hz[illustrative_frequency_mask] / 1e6,
        output_spectrum_amplitude[illustrative_frequency_mask],
        c="C1",
    )
    _ax["output_frequency"].set_ylim(0, spectrum_amplitude_limit)
    _ax["output_frequency"].set_xlabel("Frequency (MHz)")
    _ax["output_frequency"].set_ylabel("Amplitude")
    _fig.tight_layout(rect=[0, 0, 1, 0.95], pad=4, h_pad=5, w_pad=3)
    mo.vstack(
        [_fig, mo.Html("<div style='height: 24px'></div>")],
        gap=0,
    )
    return


if __name__ == "__main__":
    app.run()
