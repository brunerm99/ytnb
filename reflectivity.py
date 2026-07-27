# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.14",
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reflectivity

    Reflectivity is one of the most important concepts to understand for interpreting weather radar data,
    because it's often the first product you'll look at.
    But, despite it seeming relatively simple, there's a lot going on to get from what the radar's receiver measures, $P_r$, to the clean product, $Z$.

    $$
    \LARGE Z = P_r \frac{(4 \pi)^3 L}{P_t G^2} \frac{2}{c \tau \Omega} \frac{\lambda^2 R^2}{\pi^5 |K|^2}\,10^{18}
    $$

    This is a new notebook type I'm trying. You'll use it in basically the same way - it's still just python -
    but marimo adds some cool interaction features and nice formatting.

    > Provided as supplemental material for [Marshall Bruner](https://tinyurl.com/marshall-bruner-yt)'s
    > [animated introduction to weather radar reflectivity](https://tinyurl.com/reflectivity-video).

    - [GitHub](https://tinyurl.com/github-nb)
    - [YouTube](https://tinyurl.com/marshall-bruner-yt-nb)
    """)
    return


@app.cell
def _():
    import gzip
    import struct
    import sys

    import marimo as mo
    import numpy as np
    from scipy import constants
    from scipy.special import spherical_jn, spherical_yn
    import matplotlib.pyplot as plt

    plt.style.use("default")
    return (
        constants,
        gzip,
        mo,
        np,
        plt,
        spherical_jn,
        spherical_yn,
        struct,
        sys,
    )


@app.cell
def weather_radar_controls(mo):
    hydrometeor_presets = {  # approx
        "Drizzle": 5.0,
        "Light rain": 20.0,
        "Moderate rain": 35.0,
        "Heavy rain": 50.0,
        "Dry snow": 15.0,
        "Wet snow": 35.0,
        "Hail": 60.0,
    }

    hydrometeor = mo.ui.radio(
        options=list(hydrometeor_presets),
        value="Moderate rain",
        inline=True,
        label="Hydrometeor preset",
    )
    frequency_ghz = mo.ui.slider(
        2.0,
        12.0,
        step=0.1,
        value=5.6,
        show_value=True,
        label="Frequency (GHz)",
    )
    transmit_power_kw = mo.ui.slider(
        10,
        1000,
        step=10,
        value=250,
        show_value=True,
        label="Peak transmit power (kW)",
    )
    antenna_gain_dbi = mo.ui.slider(
        25,
        50,
        step=0.5,
        value=42,
        show_value=True,
        label="Antenna gain (dBi)",
    )
    beamwidth_deg = mo.ui.slider(
        0.3,
        3.0,
        step=0.1,
        value=1.0,
        show_value=True,
        label="One-way 3 dB beamwidth (deg)",
    )
    pulse_width_us = mo.ui.slider(
        0.1,
        5.0,
        step=0.1,
        value=1.0,
        show_value=True,
        label="Pulse width (µs)",
    )
    system_loss_db = mo.ui.slider(
        0,
        12,
        step=0.5,
        value=4,
        show_value=True,
        label="System loss (dB)",
    )
    noise_figure_db = mo.ui.slider(
        1,
        12,
        step=0.5,
        value=4,
        show_value=True,
        label="Receiver noise figure (dB)",
    )
    minimum_snr_db = mo.ui.slider(
        -5,
        20,
        step=1,
        value=3,
        show_value=True,
        label="Required SNR (dB)",
    )
    max_range_km = mo.ui.slider(
        25,
        300,
        step=5,
        value=200,
        show_value=True,
        label="Maximum plotted range (km)",
    )
    return (
        antenna_gain_dbi,
        beamwidth_deg,
        frequency_ghz,
        hydrometeor,
        hydrometeor_presets,
        max_range_km,
        minimum_snr_db,
        noise_figure_db,
        pulse_width_us,
        system_loss_db,
        transmit_power_kw,
    )


@app.cell
def weather_radar_model(
    antenna_gain_dbi,
    beamwidth_deg,
    constants,
    frequency_ghz,
    hydrometeor,
    hydrometeor_presets,
    max_range_km,
    minimum_snr_db,
    noise_figure_db,
    np,
    pulse_width_us,
    system_loss_db,
    transmit_power_kw,
):
    range_km = np.linspace(1.0, max_range_km.value, 600)
    selected_target_dbz = hydrometeor_presets[hydrometeor.value]

    assumed_receiver_temperature_k = 290.0
    liquid_water_dielectric_factor = 0.93
    wavelength_m = constants.speed_of_light / (frequency_ghz.value * 1e9)
    peak_power_w = transmit_power_kw.value * 1e3
    antenna_gain_linear = 10 ** (antenna_gain_dbi.value / 10)
    beamwidth_rad = np.deg2rad(beamwidth_deg.value)
    pulse_width_s = pulse_width_us.value * 1e-6
    system_loss_linear = 10 ** (system_loss_db.value / 10)
    noise_factor_linear = 10 ** (noise_figure_db.value / 10)
    required_snr_linear = 10 ** (minimum_snr_db.value / 10)
    matched_filter_bandwidth_hz = 1 / (2 * pulse_width_s)
    range_m = range_km * 1e3

    noise_power_w = (
        constants.Boltzmann
        * assumed_receiver_temperature_k
        * matched_filter_bandwidth_hz
        * noise_factor_linear
    )
    receiver_threshold_w = noise_power_w * required_snr_linear
    receiver_threshold_dbm = 10 * np.log10(receiver_threshold_w * 1e3)

    power_per_z_w = (
        peak_power_w
        * antenna_gain_linear**2
        * np.pi**3
        * liquid_water_dielectric_factor
        * beamwidth_rad**2
        * constants.speed_of_light
        * pulse_width_s
        * 1e-18
        / (
            1024
            * np.log(2)
            * wavelength_m**2
            * range_m**2
            * system_loss_linear
        )
    )

    minimum_detectable_z = receiver_threshold_w / power_per_z_w
    minimum_detectable_dbz = 10 * np.log10(minimum_detectable_z)
    target_z_linear = 10 ** (selected_target_dbz / 10)
    received_power_w = power_per_z_w * target_z_linear
    received_power_dbm = 10 * np.log10(received_power_w * 1e3)

    detectable_mask = minimum_detectable_dbz <= selected_target_dbz
    detection_range_km = (
        float(range_km[np.flatnonzero(detectable_mask)[-1]])
        if np.any(detectable_mask)
        else 0.0
    )
    return (
        detectable_mask,
        detection_range_km,
        minimum_detectable_dbz,
        range_km,
        received_power_dbm,
        receiver_threshold_dbm,
        selected_target_dbz,
    )


@app.cell
def weather_radar_plot(
    antenna_gain_dbi,
    beamwidth_deg,
    detectable_mask,
    detection_range_km,
    frequency_ghz,
    hydrometeor,
    max_range_km,
    minimum_detectable_dbz,
    minimum_snr_db,
    mo,
    noise_figure_db,
    plt,
    pulse_width_us,
    range_km,
    received_power_dbm,
    receiver_threshold_dbm,
    selected_target_dbz,
    system_loss_db,
    transmit_power_kw,
):
    fig_reflectivity, axes_reflectivity = plt.subplots(
        1, 2, figsize=(12.5, 5.2), constrained_layout=True
    )
    ax_reflectivity, ax_power = axes_reflectivity

    ax_reflectivity.plot(
        range_km,
        minimum_detectable_dbz,
        color="C0",
        lw=2.2,
        label="Minimum detectable reflectivity",
    )
    ax_reflectivity.axhline(
        selected_target_dbz,
        color="C3",
        ls="--",
        lw=2,
        label=f"{hydrometeor.value}: {selected_target_dbz:.0f} dBZ",
    )
    ax_reflectivity.fill_between(
        range_km,
        minimum_detectable_dbz,
        selected_target_dbz,
        where=detectable_mask,
        color="C2",
        alpha=0.18,
        label="Detectable",
    )
    ax_reflectivity.set(
        xlabel="Range (km)",
        ylabel="Reflectivity (dBZ)",
        title="Reflectivity needed for detection",
        xlim=(1, max_range_km.value),
    )
    ax_reflectivity.grid(which="both", alpha=0.25)
    ax_reflectivity.legend(fontsize=8)

    ax_power.plot(
        range_km,
        received_power_dbm,
        color="C1",
        lw=2.2,
        label=f"Return from {hydrometeor.value.lower()}",
    )
    ax_power.axhline(
        receiver_threshold_dbm,
        color="black",
        ls="--",
        lw=1.7,
        label=f"Receiver threshold: {receiver_threshold_dbm:.1f} dBm",
    )
    ax_power.fill_between(
        range_km,
        received_power_dbm,
        receiver_threshold_dbm,
        where=detectable_mask,
        color="C2",
        alpha=0.18,
    )
    ax_power.set(
        xlabel="Range (km)",
        ylabel="Received power (dBm)",
        title="Selected target return",
        xlim=(1, max_range_km.value),
    )
    ax_power.grid(which="both", alpha=0.25)
    ax_power.legend(fontsize=8)

    if detection_range_km > 0:
        ax_reflectivity.axvline(detection_range_km, color="C2", ls=":", lw=1.8)
        ax_power.axvline(detection_range_km, color="C2", ls=":", lw=1.8)

    detection_range_label = (
        f"about **{detection_range_km:.0f} km**"
        if detection_range_km < max_range_km.value - 0.5
        else f"at least **{max_range_km.value:.0f} km** (plot limit)"
    )

    mo.vstack(
        [
            "A common task when designing a radar is to check which types and intensities of weather your radar will be able to observe."
            "Use the inputs below to try out different receiver and weather configurations "
            "and build an intuition for how design decisions affect detection.",
            mo.md(
                r"""To get minimum detectable reflectivity, basically just take the reflectivity equation we found, and replace the received power, with the minimum received power you would see in a given scenario:

                $$ \LARGE Z_{\min} = P_{r,\min} \frac{(4 \pi)^3 L}{P_t G^2} \frac{2}{c \tau \Omega} \frac{\lambda^2 R^2}{\pi^5 |K|^2}\,10^{18} $$ """,
            ),
            hydrometeor,
            mo.hstack(
                [
                    mo.vstack(
                        [
                            frequency_ghz,
                            transmit_power_kw,
                            antenna_gain_dbi,
                            beamwidth_deg,
                        ]
                    ),
                    mo.vstack(
                        [
                            pulse_width_us,
                            system_loss_db,
                            noise_figure_db,
                            minimum_snr_db,
                            max_range_km,
                        ]
                    ),
                ],
                widths="equal",
                gap=2,
            ),
            mo.md(f"""
        **{hydrometeor.value}: {selected_target_dbz:.0f} dBZ**

        <small>This is a simplistic clear-air sensitivity model. Doesn't account for attenuation, beam blockage, pulse compression, coherent integration, range weighting, and non-Rayleigh scattering.</small>
        """),
            fig_reflectivity,
        ]
    )
    return


@app.cell
def _(mo):
    nexrad_scan_keys = {
        "22:56 UTC": ("2013/05/31/KTLX/KTLX20130531_225611_V06.gz"),
        "23:05 UTC": ("2013/05/31/KTLX/KTLX20130531_230523_V06.gz"),
    }

    nexrad_time = mo.ui.dropdown(
        options=list(nexrad_scan_keys),
        value="22:56 UTC",
        label="KTLX scan time",
    )
    return nexrad_scan_keys, nexrad_time


@app.cell
async def nexrad_reader(gzip, mo, nexrad_scan_keys, np, struct, sys):
    async def fetch_binary(url):
        if sys.platform == "emscripten":
            from pyodide.http import pyfetch

            response = await pyfetch(url)
            response.raise_for_status()
            return await response.bytes()

        from urllib.request import urlopen

        with urlopen(url) as response:
            return response.read()

    def read_nexrad_level2(file_bytes, max_range_km=200):
        if not file_bytes.startswith(b"\x1f\x8b"):
            raise ValueError(
                "Expected a gzip-compressed NEXRAD Archive-II file"
            )

        archive_bytes = gzip.decompress(file_bytes)
        if archive_bytes[28:30] not in (b"\x00\x00", b"\x09\x80"):
            raise ValueError(
                "This compact reader expects uncompressed Archive-II records"
            )

        records = memoryview(archive_bytes)[36:]
        selected_rays = {
            "REF": {"sweep_number": None, "rays": []},
            "VEL": {"sweep_number": None, "rays": []},
        }
        position = 0
        message_header_size = 16
        record_size = 2432

        while position + message_header_size <= len(records):
            size_halfwords, unused_channels, message_type = struct.unpack_from(
                ">HBB", records, position
            )
            if message_type != 31:
                position += record_size
                continue

            message_size = size_halfwords * 2 - 4
            message_start = position + message_header_size
            next_position = message_start + message_size
            if message_size < 68 or next_position > len(records):
                break

            message = records[message_start:next_position]
            azimuth_deg = struct.unpack_from(">f", message, 12)[0]
            elevation_number = message[22]
            elevation_deg = struct.unpack_from(">f", message, 24)[0]
            available_pointers = max(0, (len(message) - 32) // 4)
            block_count = min(
                struct.unpack_from(">H", message, 30)[0],
                10,
                available_pointers,
            )
            block_pointers = struct.unpack_from(
                f">{block_count}I", message, 32
            )

            for block_pointer in block_pointers:
                if block_pointer == 0 or block_pointer + 28 > len(message):
                    continue

                moment = (
                    bytes(message[block_pointer + 1 : block_pointer + 4])
                    .decode("ascii")
                    .strip()
                )
                if moment not in selected_rays:
                    continue

                moment_group = selected_rays[moment]
                selected_sweep_number = moment_group["sweep_number"]
                if (
                    selected_sweep_number is not None
                    and elevation_number > selected_sweep_number
                ):
                    continue
                if (
                    selected_sweep_number is None
                    or elevation_number < selected_sweep_number
                ):
                    moment_group["sweep_number"] = elevation_number
                    moment_group["rays"] = []

                ngates = struct.unpack_from(">H", message, block_pointer + 8)[
                    0
                ]
                first_gate_m = struct.unpack_from(
                    ">h", message, block_pointer + 10
                )[0]
                gate_spacing_m = struct.unpack_from(
                    ">h", message, block_pointer + 12
                )[0]
                word_size = message[block_pointer + 19]
                scale, offset = struct.unpack_from(
                    ">ff", message, block_pointer + 20
                )
                if word_size not in (8, 16) or gate_spacing_m <= 0:
                    continue

                range_limited_gates = max(
                    0,
                    int((max_range_km * 1000 - first_gate_m) / gate_spacing_m)
                    + 1,
                )
                gates_to_read = min(ngates, range_limited_gates)
                bytes_per_gate = word_size // 8
                data_start = block_pointer + 28
                data_end = data_start + gates_to_read * bytes_per_gate
                if gates_to_read == 0 or data_end > len(message):
                    continue

                raw_values = np.frombuffer(
                    message,
                    dtype=">u2" if word_size == 16 else ">u1",
                    count=gates_to_read,
                    offset=data_start,
                )
                values = (raw_values.astype(np.float32) - offset) / scale
                values[raw_values <= 1] = np.nan
                moment_group["rays"].append(
                    (
                        azimuth_deg,
                        elevation_deg,
                        first_gate_m,
                        gate_spacing_m,
                        values,
                    )
                )

            position = next_position

        sweeps = {}
        for moment, moment_group in selected_rays.items():
            rays = sorted(moment_group["rays"], key=lambda ray: ray[0])
            if not rays:
                raise ValueError(f"No {moment} rays found in NEXRAD file")

            gate_count = max(len(ray[4]) for ray in rays)
            data = np.full((len(rays), gate_count), np.nan, dtype=np.float32)
            for ray_index, ray in enumerate(rays):
                data[ray_index, : len(ray[4])] = ray[4]

            sweeps[moment] = {
                "azimuth_deg": np.array(
                    [ray[0] for ray in rays], dtype=np.float32
                ),
                "elevation_deg": float(np.mean([ray[1] for ray in rays])),
                "first_gate_m": rays[0][2],
                "gate_spacing_m": rays[0][3],
                "data": data,
            }

        return sweeps

    async def load_nexrad_scans(scan_keys):
        s3_root = "https://unidata-nexrad-level2.s3.amazonaws.com"
        sweeps_by_time = {}
        for scan_label, s3_key in scan_keys.items():
            file_bytes = await fetch_binary(f"{s3_root}/{s3_key}")
            sweeps_by_time[scan_label] = read_nexrad_level2(file_bytes)
        return sweeps_by_time

    with mo.status.spinner("Downloading and decoding two NEXRAD scans…"):
        nexrad_sweeps_by_time = await load_nexrad_scans(nexrad_scan_keys)
    return (nexrad_sweeps_by_time,)


@app.cell
def nexrad_plot(mo, nexrad_sweeps_by_time, nexrad_time, np, plt):
    def make_nexrad_plot(sweeps, scan_label):
        figure, axes = plt.subplots(
            1, 2, figsize=(12.5, 5.6), constrained_layout=True
        )
        plot_specs = [
            ("REF", axes[0], "turbo", -10, 75, "Reflectivity (dBZ)"),
            ("VEL", axes[1], "RdBu_r", -40, 40, "Velocity (m/s)"),
        ]

        for moment, axis, colormap, lower, upper, colorbar_label in plot_specs:
            sweep = sweeps[moment]
            azimuth_centers = np.unwrap(np.deg2rad(sweep["azimuth_deg"]))
            azimuth_midpoints = (
                azimuth_centers[:-1] + azimuth_centers[1:]
            ) / 2
            azimuth_edges = np.concatenate(
                [
                    [
                        azimuth_centers[0]
                        - (azimuth_midpoints[0] - azimuth_centers[0])
                    ],
                    azimuth_midpoints,
                    [
                        azimuth_centers[-1]
                        + (azimuth_centers[-1] - azimuth_midpoints[-1])
                    ],
                ]
            )
            range_edges_km = (
                sweep["first_gate_m"]
                - sweep["gate_spacing_m"] / 2
                + np.arange(sweep["data"].shape[1] + 1)
                * sweep["gate_spacing_m"]
            ) / 1000
            range_edges_km = np.maximum(range_edges_km, 0)
            x_edges_km = np.sin(azimuth_edges)[:, None] * range_edges_km
            y_edges_km = np.cos(azimuth_edges)[:, None] * range_edges_km

            image = axis.pcolormesh(
                x_edges_km,
                y_edges_km,
                sweep["data"],
                cmap=colormap,
                vmin=lower,
                vmax=upper,
                shading="flat",
                rasterized=True,
            )
            colorbar = figure.colorbar(image, ax=axis, pad=0.02)
            colorbar.set_label(colorbar_label)
            axis.set(
                xlim=(-130, 30),
                ylim=(-30, 130),
                aspect="equal",
                xlabel="East–west distance from KTLX (km)",
                ylabel="North–south distance from KTLX (km)",
                title=(
                    f"{'Base reflectivity' if moment == 'REF' else 'Radial velocity'}"
                    f" · {scan_label}"
                ),
            )
            axis.grid(alpha=0.2)

        return figure

    selected_nexrad_time = nexrad_time.value
    selected_nexrad_sweeps = nexrad_sweeps_by_time[selected_nexrad_time]
    nexrad_figure = make_nexrad_plot(
        selected_nexrad_sweeps, selected_nexrad_time
    )

    mo.vstack(
        [
            mo.md(r"""
        ---

        ## El Reno tornado in NEXRAD Level II data
        """),
            nexrad_time,
            mo.md(f"""
        """),
            nexrad_figure,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Backscatter

    This next part will be important for the next video which goes into more of the physics of reflectivity.
    """)
    return


@app.cell
def _(np, spherical_jn, spherical_yn):
    def pec_sphere_backscatter(size_parameter):
        """Monostatic RCS of a PEC sphere, normalized by its projected area."""
        values = np.atleast_1d(np.asarray(size_parameter, dtype=float))
        if np.any(values <= 0):
            raise ValueError("size_parameter must be positive")

        result = np.empty_like(values)
        for index, x in enumerate(values.flat):
            # Standard Mie-series cutoff for size parameter x = ka.
            orders = np.arange(1, int(np.ceil(x + 4 * x ** (1 / 3) + 2)) + 1)
            jn = spherical_jn(orders, x)
            yn = spherical_yn(orders, x)
            jn_prime = spherical_jn(orders, x, derivative=True)
            yn_prime = spherical_yn(orders, x, derivative=True)

            psi = x * jn
            xi = x * (jn + 1j * yn)
            psi_prime = jn + x * jn_prime
            xi_prime = psi_prime + 1j * (yn + x * yn_prime)
            a_n = -psi_prime / xi_prime
            b_n = -psi / xi

            amplitude = np.sum((2 * orders + 1) * (-1) ** orders * (a_n - b_n))
            result.flat[index] = abs(amplitude) ** 2 / x**2

        return result.item() if np.ndim(size_parameter) == 0 else result

    return (pec_sphere_backscatter,)


@app.cell
def pec_sphere_curve(np, pec_sphere_backscatter):
    ka = np.logspace(-1, 2, 2000)
    normalized_rcs = pec_sphere_backscatter(ka)
    return ka, normalized_rcs


@app.cell
def pec_sphere_controls(mo):
    rcs_frequency_ghz = mo.ui.slider(
        1.0,
        10.0,
        step=0.1,
        value=5.6,
        show_value=True,
        label="Frequency (GHz)",
    )
    sphere_diameter_cm = mo.ui.slider(
        1.0,
        90.0,
        step=1.0,
        value=10.0,
        show_value=True,
        label="Sphere diameter (cm)",
    )
    return rcs_frequency_ghz, sphere_diameter_cm


@app.cell
def _(
    constants,
    ka,
    mo,
    normalized_rcs,
    np,
    pec_sphere_backscatter,
    plt,
    rcs_frequency_ghz,
    sphere_diameter_cm,
):
    sphere_radius_m = sphere_diameter_cm.value / 200
    rcs_wavelength_m = constants.speed_of_light / (
        rcs_frequency_ghz.value * 1e9
    )
    selected_ka = 2 * np.pi * sphere_radius_m / rcs_wavelength_m
    selected_normalized_rcs = pec_sphere_backscatter(selected_ka)
    selected_rcs_m2 = selected_normalized_rcs * np.pi * sphere_radius_m**2
    selected_rcs_dbsm = 10 * np.log10(selected_rcs_m2)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(
        ka, normalized_rcs, color="black", lw=1.6, label="exact Mie series"
    )
    ax.loglog(ka, 9 * ka**4, "--", lw=1.2, label=r"Rayleigh: $9(ka)^4$")
    # ax.axhline(1, color="C3", ls="--", lw=1.2, label="optical limit: 1")
    ax.scatter(
        [selected_ka],
        [selected_normalized_rcs],
        s=95,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label="selected sphere",
    )
    ax.annotate(
        f"ka = {selected_ka:.2f}",
        xy=(selected_ka, selected_normalized_rcs),
        xytext=(9, 9),
        textcoords="offset points",
        fontsize=9,
        weight="bold",
    )
    ax.set(
        xlim=(0.1, 100),
        ylim=(5e-4, 5),
        xlabel=r"Size parameter $ka = 2\pi a / \lambda$",
        ylabel=r"Normalized monostatic RCS $\sigma / (\pi a^2)$",
        title="Radar cross section of a perfectly conducting sphere",
    )
    ax.grid(which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    mo.vstack(
        [
            mo.hstack(
                [rcs_frequency_ghz, sphere_diameter_cm],
                widths="equal",
                gap=2,
            ),
            mo.md(f"""
        **Selected point:** {rcs_frequency_ghz.value:.1f} GHz, {sphere_diameter_cm.value:.0f} cm diameter<br>
        Wavelength: {rcs_wavelength_m * 100:.2f} cm · $ka$: {selected_ka:.3f} · Normalized RCS: {selected_normalized_rcs:.3g}<br>
        Absolute RCS: {selected_rcs_m2:.3g} m² ({selected_rcs_dbsm:.1f} dBsm)
        """),
            mo.mpl.interactive(ax),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
