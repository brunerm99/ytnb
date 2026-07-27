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
    \LARGE Z = P_r \frac{(4 \pi)^3 L}{P_t G^2} \frac{2}{c \tau \Omega} \frac{\lambda^2 R^2}{\pi^5 |K|^2}
    $$

    This is a new notebook type I'm trying. You'll use it in basically the same way - it's still just python -
    but marimo adds some cool interaction features, nice formatting,
    and an app mode if you don't care about the code and just want to play around with the concepts (`Ctrl+.`).

    > Provided as supplemental material for [Marshall Bruner](https://tinyurl.com/marshall-bruner-yt)'s
    > [animated introduction to dual-pol weather radar](https://tinyurl.com/reflectivity-video).

    - [GitHub](https://tinyurl.com/github-nb)
    - [YouTube](https://tinyurl.com/marshall-bruner-yt-nb)

    ![thumbnail](https://drive.google.com/uc?id=1e3Lxy-2z0uBRiAc386v0DHbcycVSeSeU)
    """)
    return


@app.cell
def _():
    from pathlib import Path
    from urllib.request import urlretrieve
    import marimo as mo
    import numpy as np
    from scipy import constants
    from scipy.special import spherical_jn, spherical_yn
    import matplotlib.pyplot as plt

    plt.style.use("default")
    import pyart

    return (
        Path,
        constants,
        mo,
        np,
        plt,
        pyart,
        spherical_jn,
        spherical_yn,
        urlretrieve,
    )


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

    mo.hstack(
        [rcs_frequency_ghz, sphere_diameter_cm],
        widths="equal",
        gap=2,
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
            mo.md(f"""
        **Selected point:** {rcs_frequency_ghz.value:.1f} GHz, {sphere_diameter_cm.value:.0f} cm diameter<br>
        Wavelength: {rcs_wavelength_m * 100:.2f} cm · $ka$: {selected_ka:.3f} · Normalized RCS: {selected_normalized_rcs:.3g}<br>
        Absolute RCS: {selected_rcs_m2:.3g} m² ({selected_rcs_dbsm:.1f} dBsm)
        """),
            mo.mpl.interactive(ax),
        ]
    )
    return


@app.cell
def weather_radar_header(mo):
    mo.md(r"""
    ---

    ## Interactive weather-radar range sensitivity

    Choose a hydrometeor preset, then adjust the radar and receiver controls. The left plot shows the **minimum detectable reflectivity** versus range; the right plot shows the return from the selected target against the receiver threshold.
    """)
    return


@app.cell
def weather_radar_controls(mo):
    hydrometeor_presets = {
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

    mo.vstack(
        [
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
        ]
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
    # Simplified monostatic weather-radar equation for a Gaussian beam.
    # Z is in mm^6 m^-3. Receiver temperature and the liquid-water
    # dielectric factor are explicit model assumptions rather than constants.
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
    detectable_mask,
    detection_range_km,
    hydrometeor,
    max_range_km,
    minimum_detectable_dbz,
    mo,
    plt,
    range_km,
    received_power_dbm,
    receiver_threshold_dbm,
    selected_target_dbz,
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
            mo.md(f"""
        ### {hydrometeor.value}: {selected_target_dbz:.0f} dBZ

        Estimated detection range: {detection_range_label}. Frequency changes wavelength sensitivity; power and gain raise the return; receiver noise, required SNR, and system loss raise the detection threshold.

        <small>This is a first-order clear-air sensitivity model. It omits attenuation, beam blockage, pulse compression, coherent integration, range weighting, and non-Rayleigh scattering.</small>
        """),
            fig_reflectivity,
        ]
    )
    return


@app.cell
def _(mo):
    nexrad_scan_keys = {
        "22:56 UTC — just before tornado formation": (
            "2013/05/31/KTLX/KTLX20130531_225611_V06.gz"
        ),
        "23:05 UTC — near peak tornado width": (
            "2013/05/31/KTLX/KTLX20130531_230523_V06.gz"
        ),
    }

    nexrad_time = mo.ui.dropdown(
        options=list(nexrad_scan_keys),
        value="22:56 UTC — just before tornado formation",
        label="KTLX scan time",
    )

    mo.vstack(
        [
            mo.md(r"""
        ---

        ## El Reno tornado in NEXRAD Level II data

        The first run downloads two KTLX volume scans from the public
        `unidata-nexrad-level2` S3 archive. Choose which time drives the plot.
        """),
            nexrad_time,
        ]
    )
    return nexrad_scan_keys, nexrad_time


@app.cell
def _(Path, mo, nexrad_scan_keys, nexrad_time, plt, pyart, urlretrieve):
    def download_nexrad_scans(scan_keys, cache_directory):
        """Download missing NEXRAD scans once and return their local paths."""
        cache_directory.mkdir(parents=True, exist_ok=True)
        downloaded_paths = {}
        s3_root = "https://unidata-nexrad-level2.s3.amazonaws.com"

        for scan_label, s3_key in scan_keys.items():
            scan_path = cache_directory / Path(s3_key).name
            if not scan_path.exists():
                urlretrieve(f"{s3_root}/{s3_key}", scan_path)
            downloaded_paths[scan_label] = scan_path

        return downloaded_paths

    def make_nexrad_plot(scan_path, scan_label):
        """Plot the lowest-sweep reflectivity and radial velocity."""
        radar_volume = pyart.io.read_nexrad_archive(str(scan_path))
        radar_display = pyart.graph.RadarDisplay(radar_volume)
        figure, axes = plt.subplots(
            1, 2, figsize=(12.5, 5.6), constrained_layout=True
        )

        radar_display.plot_ppi(
            "reflectivity",
            sweep=0,
            ax=axes[0],
            cmap="NWSRef",
            vmin=-10,
            vmax=75,
            title=f"Base reflectivity · {scan_label}",
            colorbar_label="Reflectivity (dBZ)",
        )
        radar_display.plot_ppi(
            "velocity",
            sweep=1,
            ax=axes[1],
            cmap="NWSVel",
            vmin=-40,
            vmax=40,
            title=f"Radial velocity · {scan_label}",
            colorbar_label="Velocity (m/s)",
        )

        for axis in axes:
            axis.set_xlim(-130, 30)
            axis.set_ylim(-30, 130)
            axis.set_aspect("equal")
            axis.grid(alpha=0.2)

        return figure, radar_volume

    nexrad_cache_directory = Path("data/nexrad")
    nexrad_local_paths = download_nexrad_scans(
        nexrad_scan_keys, nexrad_cache_directory
    )
    selected_nexrad_time = nexrad_time.value
    selected_nexrad_path = nexrad_local_paths[selected_nexrad_time]
    nexrad_figure, selected_radar_volume = make_nexrad_plot(
        selected_nexrad_path, selected_nexrad_time
    )

    mo.vstack(
        [
            mo.md(f"""
        **KTLX {selected_nexrad_time}** · lowest elevation sweep

        Reflectivity shows storm structure; velocity shows motion toward and away
        from the radar. Coordinates are kilometers east and north of KTLX.
        """),
            nexrad_figure,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
