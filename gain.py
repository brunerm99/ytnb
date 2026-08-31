# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scikit-rf",
#     "anywidget",
#     "traitlets",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


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


@app.cell(hide_code=True)
def _(mo):
    mo.sidebar(
        mo.outline(label="Contents"),
        width="18rem",
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import skrf as rf
    import anywidget
    import traitlets

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
    return anywidget, np, plt, rf, traitlets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decibel (dB) Scale

    Voltage gain:

    $$
        \Large G = 20 \log_{10}{\left(\frac{V_{out}}{V_{in}}\right)}
    $$

    Power gain:

    $$
        \Large G = 10 \log_{10}{\left(\frac{P_{out}}{P_{in}}\right)}
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
    # N = 2**12
    N = illustrative_sample_count
    illustrative_fft_frequency_hz = np.fft.rfftfreq(
        N,
        1 / illustrative_sample_rate_hz,
    )
    input_spectrum = np.fft.rfft(input_signal, N)
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
        N,
        component_frequencies_hz,
        illustrative_fft_frequency_hz,
        illustrative_frequency_mask,
        illustrative_time_s,
        input_signal,
        input_spectrum,
        input_spectrum_amplitude,
    )


@app.cell
def _(anywidget, component_frequencies_hz, mo, traitlets):
    _EQ_ESM = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg";
      const W = 760, H = 400, ML = 64, MR = 18, MT = 46, MB = 58;
      const XMIN = 0, XMAX = 40, YMIN = -22, YMAX = 16;
      const GMIN = -20, GMAX = 15, SNAP = 0.5;

      const px = (f) => ML + ((f - XMIN) / (XMAX - XMIN)) * (W - ML - MR);
      const py = (g) => MT + ((YMAX - g) / (YMAX - YMIN)) * (H - MT - MB);

      const freqs = model.get("freqs_mhz");
      let gains = [...model.get("gains_db")];
      const markers = model.get("markers_mhz");
      const controller = new AbortController();
      const { signal } = controller;

      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.style.cssText =
        "width:100%;max-width:860px;display:block;margin:0 auto;user-select:none;" +
        "touch-action:none;background:#fff;border-radius:4px;" +
        "font-family:'DejaVu Sans',Verdana,sans-serif;";

      const make = (tag, attrs, parent = svg) => {
        const n = document.createElementNS(NS, tag);
        for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
        parent.append(n);
        return n;
      };
      const text = (str, attrs, parent = svg) => {
        const t = make("text", attrs, parent);
        t.textContent = str;
        return t;
      };

      make("rect", { x: ML, y: MT, width: W - ML - MR, height: H - MT - MB, fill: "#eaeaf2" });
      for (let f = XMIN; f <= XMAX; f += 5) {
        make("line", { x1: px(f), y1: MT, x2: px(f), y2: H - MB, stroke: "#fff", "stroke-width": 1.2 });
        text(String(f), { x: px(f), y: H - MB + 20, "text-anchor": "middle", "font-size": 13, fill: "#333" });
      }
      for (let g = -20; g <= 15; g += 5) {
        make("line", { x1: ML, y1: py(g), x2: W - MR, y2: py(g), stroke: "#fff", "stroke-width": 1.2 });
        text(String(g), { x: ML - 8, y: py(g) + 4, "text-anchor": "end", "font-size": 13, fill: "#333" });
      }
      make("line", { x1: ML, y1: py(0), x2: W - MR, y2: py(0), stroke: "#000", "stroke-width": 1 });
      for (const mf of markers) {
        if (mf < XMIN || mf > XMAX) continue;
        make("line", {
          x1: px(mf), y1: MT, x2: px(mf), y2: H - MB,
          stroke: "#c44e52", "stroke-width": 1, "stroke-dasharray": "4 4", opacity: 0.55,
        });
      }
      text("Dummy Values Example", {
        x: (ML + W - MR) / 2, y: 26, "text-anchor": "middle",
        "font-size": 19, "font-weight": 600, fill: "#222",
      });
      text("Frequency (MHz)", {
        x: (ML + W - MR) / 2, y: H - 14, "text-anchor": "middle", "font-size": 15, fill: "#222",
      });
      const yl = text("Gain (dB)", {
        x: 18, y: (MT + H - MB) / 2, "text-anchor": "middle", "font-size": 15, fill: "#222",
      });
      yl.setAttribute("transform", `rotate(-90 18 ${(MT + H - MB) / 2})`);

      const curve = make("polyline", {
        fill: "none", stroke: "#4c72b0", "stroke-width": 2.2, "stroke-linejoin": "round",
      });

      const editable = [];
      freqs.forEach((f, i) => {
        if (f >= XMIN && f <= XMAX) editable.push(i);
      });
      const handles = editable.map(() =>
        make("circle", {
          r: 7, fill: "#c44e52", stroke: "#fff", "stroke-width": 1.5, cursor: "ns-resize",
        })
      );
      const label = text("", {
        x: 0, y: 0, "text-anchor": "middle", "font-size": 13, "font-weight": 600,
        fill: "#c44e52", visibility: "hidden",
      });

      const redraw = () => {
        curve.setAttribute(
          "points",
          editable.map((i) => `${px(freqs[i])},${py(gains[i])}`).join(" ")
        );
        editable.forEach((i, k) => {
          handles[k].setAttribute("cx", px(freqs[i]));
          handles[k].setAttribute("cy", py(gains[i]));
        });
      };

      const toGain = (clientY) => {
        const r = svg.getBoundingClientRect();
        const sy = ((clientY - r.top) * H) / r.height;
        let g = YMAX - ((sy - MT) * (YMAX - YMIN)) / (H - MT - MB);
        g = Math.max(GMIN, Math.min(GMAX, g));
        return Math.round(g / SNAP) * SNAP;
      };

      handles.forEach((h, k) => {
        const i = editable[k];
        h.addEventListener(
          "pointerdown",
          (e) => {
            e.preventDefault();
            h.setPointerCapture(e.pointerId);
            const move = (ev) => {
              gains[i] = toGain(ev.clientY);
              redraw();
              label.setAttribute("x", px(freqs[i]));
              label.setAttribute("y", py(gains[i]) - 14);
              label.setAttribute("visibility", "visible");
              label.textContent =
                `${gains[i] > 0 ? "+" : ""}${gains[i].toFixed(1)} dB`;
            };
            const up = () => {
              h.removeEventListener("pointermove", move);
              label.setAttribute("visibility", "hidden");
              model.set("gains_db", [...gains]);
              model.save_changes();
            };
            h.addEventListener("pointermove", move);
            h.addEventListener("pointerup", up, { once: true });
            h.addEventListener("pointercancel", up, { once: true });
          },
          { signal }
        );
      });

      model.on("change:gains_db", () => {
        gains = [...model.get("gains_db")];
        redraw();
      });

      redraw();
      const wrap = document.createElement("div");
      wrap.style.cssText = "display:flex;align-items:center;gap:10px;";
      const btn = document.createElement("button");
      btn.textContent = "All high";
      btn.title = "Set every band to +15 dB";
      btn.style.cssText =
        "flex:none;padding:6px 12px;border:1px solid #c44e52;border-radius:6px;" +
        "background:#fff;color:#c44e52;font-weight:600;cursor:pointer;";
      btn.addEventListener(
        "click",
        () => {
          for (const i of editable) gains[i] = GMAX;
          redraw();
          model.set("gains_db", [...gains]);
          model.save_changes();
        },
        { signal }
      );
      wrap.append(svg, btn);
      el.append(wrap);
      return () => controller.abort();
    }
    export default { render };
    """

    class _GainEQ(anywidget.AnyWidget):
        _esm = _EQ_ESM
        freqs_mhz = traitlets.List(traitlets.Float()).tag(sync=True)
        gains_db = traitlets.List(traitlets.Float()).tag(sync=True)
        markers_mhz = traitlets.List(traitlets.Float()).tag(sync=True)

    gain_eq = _GainEQ(
        freqs_mhz=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 100.0],
        gains_db=[0.0, 12.0, 8.0, 3.0, 0.0, -4.0, -9.0, -12.0, -15.0, -20.0],
        markers_mhz=[float(f) for f in component_frequencies_hz / 1e6],
    )
    get_eq_gains_db, set_eq_gains_db = mo.state(gain_eq.gains_db)
    gain_eq.observe(
        lambda _: set_eq_gains_db(gain_eq.gains_db), names=["gains_db"]
    )
    gain_eq
    return gain_eq, get_eq_gains_db


@app.cell
def _(gain_eq, get_eq_gains_db, illustrative_fft_frequency_hz, np):
    illustrative_amp_frequency_hz = np.array(gain_eq.freqs_mhz) * 1e6
    illustrative_amp_gain_db = np.array(get_eq_gains_db())
    interpolated_amp_gain_db = np.interp(
        illustrative_fft_frequency_hz,
        illustrative_amp_frequency_hz,
        illustrative_amp_gain_db,
    )
    illustrative_voltage_gain = 10 ** (interpolated_amp_gain_db / 20)
    return (illustrative_voltage_gain,)


@app.cell
def _(
    N,
    illustrative_fft_frequency_hz,
    illustrative_frequency_mask,
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
    output_signal = np.fft.irfft(output_spectrum, n=N)
    output_spectrum_amplitude = 2 * np.abs(output_spectrum) / N
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
    gain_eq,
    illustrative_fft_frequency_hz,
    illustrative_frequency_mask,
    illustrative_time_s,
    input_signal,
    input_spectrum_amplitude,
    mo,
    output_signal,
    output_spectrum_amplitude,
    plt,
    spectrum_amplitude_limit,
    time_amplitude_limit,
):
    _fig_in, _ax_in = plt.subplots(ncols=2, figsize=(12, 4))
    _ax_in[0].set_title("Input Time Series")
    _ax_in[0].plot(illustrative_time_s / 1e-6, input_signal, c="C0")
    _ax_in[0].set_ylim(-time_amplitude_limit, time_amplitude_limit)
    _ax_in[0].set_xlabel(r"Time ($\mu$s)")
    _ax_in[0].set_ylabel("Amplitude")
    _ax_in[1].set_title("Input Frequency Spectrum")
    _ax_in[1].plot(
        illustrative_fft_frequency_hz[illustrative_frequency_mask] / 1e6,
        input_spectrum_amplitude[illustrative_frequency_mask],
        c="C0",
    )
    _ax_in[1].set_ylim(0, spectrum_amplitude_limit)
    _ax_in[1].set_xlabel("Frequency (MHz)")
    _ax_in[1].set_ylabel("Amplitude")
    _fig_in.tight_layout(pad=3, w_pad=3)

    _fig_out, _ax_out = plt.subplots(ncols=2, figsize=(12, 4))
    _ax_out[0].set_title("Output Time Series")
    _ax_out[0].plot(illustrative_time_s / 1e-6, output_signal, c="C1")
    _ax_out[0].set_ylim(-time_amplitude_limit, time_amplitude_limit)
    _ax_out[0].set_xlabel(r"Time ($\mu$s)")
    _ax_out[0].set_ylabel("Amplitude")
    _ax_out[1].set_title("Output Frequency Spectrum")
    _ax_out[1].plot(
        illustrative_fft_frequency_hz[illustrative_frequency_mask] / 1e6,
        output_spectrum_amplitude[illustrative_frequency_mask],
        c="C1",
    )
    _ax_out[1].set_ylim(0, spectrum_amplitude_limit)
    _ax_out[1].set_xlabel("Frequency (MHz)")
    _ax_out[1].set_ylabel("Amplitude")
    _fig_out.tight_layout(pad=3, w_pad=3)

    mo.vstack(
        [
            mo.md("### Multi-Tone Signal Through an Example Amplifier"),
            _fig_in,
            gain_eq,
            _fig_out,
            mo.Html("<div style='height: 24px'></div>"),
        ],
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using a real component: ADL8142
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
    ### Amplifying a linear frequency-modulated (LFM) signal
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


if __name__ == "__main__":
    app.run()
