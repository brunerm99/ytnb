import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What Happens When a Radar Wave Hits a Raindrop?

    > Note that this is a marimo notebook (not google colab), so we have some extra features! The whole thing runs in your browser, so there's nothing to install and no server running the code. It comes up in "app mode", which shows you the formatted markdown and the code outputs but not the code itself. The "..." menu in the top right has a "Show code" toggle if you want to read it, and it will also let you download the notebook as a `.py` file. To actually edit things and run them yourself, download that file and open it with marimo on your own machine.

    A radar image is made up of a ton of pixels. But what are the pixels made of?

    The video explains some of the physics of how this works using animations, but here we'll use more equations and interactive elements to drive home the concept.

    When we start diving down to the lower levels of this concept, we start to encounter some more difficult math. Here's some resources that will provide more context and more in-depth discussions:
    - Intro to Dual-Pol Weather Radar (rayleigh, dipole radiation pattern, good beginner resource) - Chandra, Beauchamp, Bechini (book) - https://tinyurl.com/intro-to-dual-pol
    - Radar Meteorology: A First Course - Rauber, Nesbitt (book) - https://tinyurl.com/meteorology-a-first-course
    - Polarimetric Doppler Weather Radar (dielectric sphere polarization) - Chandra, Bringi (book) - https://tinyurl.com/polarimetric-chandra
    - Absorption and Scattering of Light by Small Particles (great resource about induced-dipole model) - Bohren (book) - https://tinyurl.com/Absorption-and-Scattering
    - Doppler Radar and Weather Observations (scattering cross-section) - Doviac, Zrnic (book) - https://tinyurl.com/doppler-radar-observations
    - The Feynman Lectures Chapter 28 (gives a good fundamental understanding, really just a classic book) - Feynman (book) - https://tinyurl.com/feynman-ch28
    - Rayleigh, Mie Scattering - Chew (paper) - https://tinyurl.com/rayleigh-mie
    - Rayleigh vs. Mie Scattering - Radartutorial (article) - https://tinyurl.com/rayleigh-mie-radartutorial
    - Dipole Antenna - AntennaTheory (article) - https://tinyurl.com/dipole-antennatheory
    - NASA NPOL OLYMPEX RHI data of Bright Band - NASA (data) - https://tinyurl.com/olympex-bright-band
    - NASA NPOL OLYMPEX all day data from Nov. 13, 2015 - NASA (data) - https://tinyurl.com/olympex-13-nov-2015
    - Radar Range Equation for Weather Radar - Radar-Tutorial (article) - https://tinyurl.com/radar-range-equation
    - The Radar Equation in Meteorology - Probert-Jones (paper) - https://tinyurl.com/probert-jones
    - Path To NEXRAD - Brown, Lewis (article) - https://tinyurl.com/path-to-nexrad
    - The WSR-88D and the WSR-88D Operational Support Facility - Crum, Alberty (article) - https://tinyurl.com/wsr88d-facility

    <img src="https://notebooks.marshallbruner.com/public/raindrops_are_antennas.png"
         alt="Raindrops are Antennas"
         style="width: 100%; max-width: 640px; border-radius: 8px; display: block; margin: 1.5rem auto 0;" />
    """)
    return


@app.cell
def imports():
    from math import gamma

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle
    from scipy.constants import epsilon_0

    return Circle, epsilon_0, gamma, mo, np, plt


@app.cell
def visual_style():
    BACKGROUND = "#183340"
    BLUE = "#58C4DD"
    ACCENT = "#FFFF00"
    GREEN = "#5CD0B3"
    INK = "#DCE6EB"
    MUTED = "#A9BBC4"
    ICE = "#DBF1FD"

    def style_axes(axis, grid=True):
        axis.set_facecolor(BACKGROUND)
        axis.figure.set_facecolor(BACKGROUND)
        if grid:
            axis.grid(color=MUTED, linewidth=0.6, alpha=0.22)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            axis.spines[side].set_color(MUTED)
        axis.tick_params(colors=MUTED)
        axis.xaxis.label.set_color(INK)
        axis.yaxis.label.set_color(INK)
        axis.title.set_color(INK)

    return ACCENT, BACKGROUND, BLUE, GREEN, ICE, INK, MUTED, style_axes


@app.cell(hide_code=True)
def notebook_outline(mo):
    mo.sidebar(mo.outline(label="Contents"), width="17rem")
    return


@app.cell
def notebook_title(mo):
    mo.md(r"""
    ## Overview of the progression

    \[
    x=ka
    \;\longrightarrow\;
    \mathbf E_{\rm in}
    \;\longrightarrow\;
    \mathbf P
    \;\longrightarrow\;
    \mathbf p
    \;\longrightarrow\;
    \mathbf E_s
    \;\longrightarrow\;
    \sigma_{\rm back}
    \;\longrightarrow\;
    Z
    \]

    Each arrow here is one piece of physics we have to do. We start with a drop sitting in the beam and we end with a number on the display.

    $x = ka$ is the size parameter, and it's the first thing we look at because it decides how hard the rest of the chain is going to be. If the drop is small compared to the wavelength, it sees a nearly uniform field and everything downstream is simple (or at least _simpler_). If it isn't, we're stuck solving the whole thing with Mie theory.

    $\mathbf{E}_{\rm in}$ is the field the drop actually sees. We're in the far field of the antenna, so it's just a plane wave washing over the particle.

    $\mathbf{P}$ is the polarization that field induces inside the drop. Water has a huge dielectric constant at microwave frequencies, so the field pulls charge apart much harder than it would in ice. This is exactly why the same size particle gives you a very different return depending on what phase it's in.

    $\mathbf{p} = \int \mathbf{P}\, dV$ collapses the whole drop down to a single point dipole. This is the step that's only really correct when $x$ is small. Get the drop big enough and different parts of it are being driven out of phase with each other, and there's no single dipole left as a simplification.

    $\mathbf{E}_s$ is what that dipole radiates back out. Accelerating charge radiates, so the oscillating dipole re-emits a field in every direction. The re-radiation is the scattering.

    You could solve for the radiation from the particle in any direction (useful for bistatic radar), but we choose $\sigma_{\rm back}$ because the direction coming back to our radar is what we care about.

    $$\sigma_{\rm back} = \lim_{r \to \infty} 4\pi r^2 \frac{|\mathbf{E}_s|^2}{|\mathbf{E}_{\rm in}|^2}$$

    It has units of area, and it's the number that tells us how visible this one drop is to a monostatic radar.

    $Z$ is where we stop talking about one drop and start talking about a volume, because a radar never sees one drop. We sum $\sigma_{\rm back}$ over the whole drop size distribution in the resolution volume. In the Rayleigh limit $\sigma_{\rm back} \propto D^6$, which is where $Z = \int N(D)\, D^6\, dD$ comes from. The $D^6$ dependence is why larger drops dominate the reflectivity.

    The reason we can walk through this in a straight line at all is that every step is linear. The field induces polarization, the polarization radiates, the radiated power adds. Once $x$ gets large enough that the dipole approximation falls apart, the chain stops being a chain and we have to solve the whole boundary value problem at once. I'll have to cover that in another video...
    """)
    return


@app.cell
def rayleigh_heading(mo):
    mo.md(r"""
    ## How small is small?

    We keep saying the drop has to be small, but small compared to what? The only length the wave knows about is its own wavelength.

    \[
    x=k_0a=\frac{\pi D}{\lambda},
    \qquad
    \Delta\phi_{\rm incident}=k_0D=2x
    \]

    The second quantity is the one that matters physically. The wave hits the front of the drop first, and by the time that crest reaches the back face the phase has advanced by $\Delta\phi = k_0 D$, which is just $2x$. So $x$ is really a measure of the phase spread across the drop, not of its size on its own.

    When $\Delta\phi$ is small, every charge inside feels about the same field at the same instant, their contributions add in phase, and the drop acts like a single dipole. When it gets large, the front is being pushed while the back is being pulled, and those contributions start cancelling. The coherence factor $|C_{\rm phase}|$ below tracks how much of the response survives.

    The rule of thumb is $D < \lambda/10$, or $x < 0.31$. At S-band that's 10 mm, and raindrops top out around 8 mm, so we're fine across the whole distribution. At X-band it's 3.2 mm, and that same 8 mm drop is already out at $x = 0.79$.

    Play with the sliders and watch $|C_{\rm phase}|$.
    """)
    return


@app.cell
def rayleigh_controls(mo):
    diameter_slider = mo.ui.slider(
        start=0.1,
        stop=8.0,
        step=0.1,
        value=1.0,
        label=r"$D$ (mm)",
        show_value=True,
    )
    wavelength_slider = mo.ui.slider(
        start=30,
        stop=110,
        step=1,
        value=100,
        label=r"$\lambda$ (mm)",
        show_value=True,
    )
    return diameter_slider, wavelength_slider


@app.cell
def rayleigh_helpers(
    ACCENT,
    BACKGROUND,
    BLUE,
    Circle,
    INK,
    MUTED,
    np,
    plt,
    style_axes,
):
    def sphere_phase_coherence(size_parameter):
        if abs(size_parameter) < 1e-4:
            return 1 - size_parameter**2 / 10
        return (
            3
            * (
                np.sin(size_parameter)
                - size_parameter * np.cos(size_parameter)
            )
            / size_parameter**3
        )

    def make_phase_figure(size_parameter):
        figure, (wave_axis, phasor_axis) = plt.subplots(
            1,
            2,
            figsize=(9.5, 3.8),
            gridspec_kw={"width_ratios": [1.45, 1]},
        )
        figure.patch.set_facecolor(BACKGROUND)

        position = np.linspace(-2.5, 2.5, 1200)
        field = 0.82 * np.cos(size_parameter * position)
        inside = np.abs(position) <= 1

        wave_axis.plot(position, field, color=BLUE, linewidth=2)
        wave_axis.plot(
            position[inside],
            field[inside],
            color=ACCENT,
            linewidth=4,
            solid_capstyle="round",
        )
        wave_axis.add_patch(
            Circle(
                (0, 0),
                1,
                facecolor=BLUE,
                edgecolor=INK,
                linewidth=1.2,
                alpha=0.18,
            )
        )
        wave_axis.axhline(0, color=MUTED, linewidth=0.8, alpha=0.5)
        wave_axis.set(
            xlim=(-2.5, 2.5),
            ylim=(-1.2, 1.2),
            xlabel=r"position / $a$",
            yticks=[],
            title="phase across the particle",
        )
        wave_axis.set_aspect("equal")
        style_axes(wave_axis, grid=False)

        slice_positions = np.linspace(-0.96, 0.96, 27)
        slice_weights = 1 - slice_positions**2
        slice_weights /= slice_weights.sum()
        slice_phases = size_parameter * slice_positions
        coherent_sum = np.sum(slice_weights * np.exp(1j * slice_phases))

        phasor_axis.set_facecolor(BACKGROUND)
        phasor_axis.add_patch(
            Circle((0, 0), 1, fill=False, edgecolor=MUTED, linewidth=1)
        )
        for phase, weight in zip(slice_phases, slice_weights):
            phasor_axis.plot(
                [0, np.cos(phase)],
                [0, np.sin(phase)],
                color=BLUE,
                linewidth=0.8 + 12 * weight,
                alpha=min(0.18 + 3 * weight, 1),
            )
        phasor_axis.arrow(
            0,
            0,
            coherent_sum.real,
            coherent_sum.imag,
            width=0.018,
            head_width=0.1,
            length_includes_head=True,
            color=ACCENT,
            zorder=5,
        )
        phasor_axis.set(
            xlim=(-1.15, 1.15),
            ylim=(-1.15, 1.15),
            xticks=[],
            yticks=[],
            title="volume-weighted phase",
        )
        phasor_axis.title.set_color(INK)
        phasor_axis.set_aspect("equal")
        for spine in phasor_axis.spines.values():
            spine.set_visible(False)

        figure.tight_layout()
        return figure

    return make_phase_figure, sphere_phase_coherence


@app.cell
def rayleigh_lab(
    diameter_slider,
    make_phase_figure,
    mo,
    np,
    plt,
    sphere_phase_coherence,
    wavelength_slider,
):
    diameter_mm = diameter_slider.value
    wavelength_mm = wavelength_slider.value
    size_parameter = np.pi * diameter_mm / wavelength_mm
    incident_phase_span_deg = np.degrees(2 * size_parameter)
    phase_coherence = abs(sphere_phase_coherence(size_parameter))

    phase_figure = make_phase_figure(size_parameter)
    phase_plot = mo.as_html(phase_figure)
    plt.close(phase_figure)

    mo.vstack(
        [
            mo.hstack(
                [diameter_slider, wavelength_slider],
                widths="equal",
                gap=2,
            ),
            mo.md(
                rf"$x={size_parameter:.3f}$"
                rf"$\qquad \Delta\phi={incident_phase_span_deg:.1f}^\circ$"
                rf"$\qquad |C_\mathrm{{phase}}|={phase_coherence:.3f}$"
            ),
            phase_plot,
        ],
        gap=1,
    )
    return diameter_mm, wavelength_mm


@app.cell
def polarization_heading(mo):
    mo.md(r"""
    ## What the field does inside the drop

    We've established the drop is small enough to see one uniform field. So what does that field do to it?

    A raindrop isn't a conductor, so the charge can't run to the surface and cancel the field the way it would on metal. The bound charge stretches a little and stops. That stretching is the polarization $\mathbf{P}$, and it sets up a field of its own pointing back against the incident field, so the drop partially shields itself.

    \[
    K=\frac{\varepsilon_r-1}{\varepsilon_r+2},
    \qquad
    \mathbf E_{\rm in}=\frac{3}{\varepsilon_r+2}\mathbf E_i,
    \qquad
    \mathbf P=3\varepsilon_0K\mathbf E_i
    \]

    Water sits near $\varepsilon_r = 80$ at microwave frequencies, so only about 4% of the incident field makes it inside (4% comes from $1-K$, compute it yourself!). That sounds like bad news for scattering, but the polarization isn't set by the internal field alone. It's that field times how hard the material responds to it, which is $\varepsilon_r-1$. Water shields well and responds strongly for the same reason, so the small factor and the large one cancel. That's how the equation for $\mathbf{E}_{i} \rightarrow \mathbf{P} = 3 \varepsilon_0 K \mathbf{E}_i$.

    What's left is $K$, and $K$ saturates. Water gives $K = 0.96$, and taking $\varepsilon_r$ to infinity only buys you the last 4%. This is the whole reason we can tell rain from snow. Dry ice is around $\varepsilon_r = 3.2$, so $K = 0.42$, and since backscatter goes as $|K|^2$ that's 0.93 against 0.18. A water sphere returns about 7 dB more than an ice sphere of the same size at the same wavelength.

    \[
    \rho_{\rm bound}=-\nabla\cdot\mathbf P=0,
    \qquad
    \sigma_{\rm bound}=\mathbf P\cdot\hat{\mathbf n},
    \qquad
    \sum q=0
    \]

    These say where the charge actually ends up. With $\mathbf{P}$ uniform there's no bound charge anywhere in the interior, so all of it collects on the surface, positive on one pole and negative on the other, falling off as the cosine of the angle from the field. The drop is still neutral overall, which is what $\sum q = 0$ is saying.

    That surface charge is the dipole we're about to integrate for.

    The sliders start at $\varepsilon_r = 3.2$, which is ice. Drag up to 80 and watch $K$ crawl the last bit to 1 while $E_{\rm in}/E_i$ collapses. Dragging the $\varepsilon_r$ slider, the exponential relationship becomes really obvious.
    """)
    return


@app.cell
def polarization_controls(mo):
    relative_permittivity_slider = mo.ui.slider(
        start=1.1,
        stop=80.0,
        step=0.1,
        value=3.2,
        label=r"lossless $\varepsilon_r$",
        show_value=True,
    )
    field_angle_slider = mo.ui.slider(
        start=0,
        stop=180,
        step=1,
        value=90,
        label=r"field angle ($^\circ$)",
        show_value=True,
    )
    field_phase_slider = mo.ui.slider(
        start=0,
        stop=1,
        step=0.01,
        value=0,
        label=r"$t/T$",
        show_value=True,
    )
    return field_angle_slider, field_phase_slider, relative_permittivity_slider


@app.cell
def polarization_helpers(ACCENT, BACKGROUND, BLUE, Circle, ICE, INK, np, plt):
    def make_polarization_figure(
        relative_permittivity,
        field_angle_deg,
        field_phase_cycles,
    ):
        figure, axis = plt.subplots(figsize=(7.8, 5.4))
        figure.patch.set_facecolor(BACKGROUND)
        axis.set_facecolor(BACKGROUND)

        angle = np.radians(field_angle_deg)
        phase_sample = np.cos(2 * np.pi * field_phase_cycles)
        field_direction = np.array([np.cos(angle), np.sin(angle)])
        material_factor = (relative_permittivity - 1) / (
            relative_permittivity + 2
        )
        polarization_sample = 3 * material_factor * phase_sample
        visual_strength = min(abs(polarization_sample) / 2.9, 1)
        polarization_direction = np.sign(polarization_sample) * field_direction

        axis.add_patch(
            Circle(
                (0, 0), 1, facecolor=BLUE, edgecolor=INK, alpha=0.18, zorder=1
            )
        )

        field_origins = np.array(
            [[x, y] for x in (-1.65, 1.65) for y in (-0.7, 0, 0.7)]
        )
        field_vectors = np.tile(0.45 * phase_sample * field_direction, (6, 1))
        axis.quiver(
            field_origins[:, 0],
            field_origins[:, 1],
            field_vectors[:, 0],
            field_vectors[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=BLUE,
            width=0.007,
        )

        pair_centers = np.array(
            [
                [x, y]
                for y in (-0.5, 0, 0.5)
                for x in (-0.5, 0, 0.5)
                if x * x + y * y < 0.6
            ]
        )
        separation = 0.13 * visual_strength
        positive_points = pair_centers + separation * polarization_direction
        negative_points = pair_centers - separation * polarization_direction
        axis.scatter(
            positive_points[:, 0],
            positive_points[:, 1],
            s=58,
            color=ACCENT,
            edgecolor=BACKGROUND,
            linewidth=0.5,
            zorder=3,
        )
        axis.scatter(
            negative_points[:, 0],
            negative_points[:, 1],
            s=58,
            color=ICE,
            edgecolor=BACKGROUND,
            linewidth=0.5,
            zorder=3,
        )

        boundary_angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        surface_charge = polarization_sample * np.cos(boundary_angles - angle)
        axis.scatter(
            np.cos(boundary_angles),
            np.sin(boundary_angles),
            s=48 * abs(surface_charge) / 2.9,
            color=[ACCENT if value >= 0 else ICE for value in surface_charge],
            zorder=4,
        )

        if visual_strength > 1e-3:
            arrow_length = 0.72 * visual_strength
            # Keep the head proportional on short arrows, otherwise the head
            # overruns the shaft and renders as a blob.
            head_width = min(0.12, 0.45 * arrow_length)
            axis.arrow(
                0,
                0,
                *(arrow_length * polarization_direction),
                width=min(0.022, 0.2 * head_width),
                head_width=head_width,
                head_length=min(0.12, 0.5 * arrow_length),
                length_includes_head=True,
                color=ACCENT,
                zorder=5,
            )
            label_point = (arrow_length + 0.13) * polarization_direction
            axis.text(
                *label_point,
                r"$\mathbf{p}$",
                color=ACCENT,
                fontsize=14,
                ha="center",
                va="center",
            )

        axis.text(-1.95, 1.18, r"$\mathbf{E}_i(t)$", color=BLUE, fontsize=13)
        axis.text(0.70, -1.25, r"$\sum q=0$", color=INK, fontsize=12)
        axis.set(
            xlim=(-2.15, 2.15),
            ylim=(-1.45, 1.45),
            xticks=[],
            yticks=[],
            title="induced polarization",
        )
        axis.title.set_color(INK)
        axis.set_aspect("equal")
        for spine in axis.spines.values():
            spine.set_visible(False)

        figure.tight_layout()
        return figure

    return (make_polarization_figure,)


@app.cell
def polarization_lab(
    epsilon_0,
    field_angle_slider,
    field_phase_slider,
    make_polarization_figure,
    mo,
    plt,
    relative_permittivity_slider,
):
    relative_permittivity = relative_permittivity_slider.value
    field_angle_deg = field_angle_slider.value
    field_phase_cycles = field_phase_slider.value

    material_factor = (relative_permittivity - 1) / (relative_permittivity + 2)
    internal_field_ratio = 3 / (relative_permittivity + 2)
    polarization_per_incident_field = 3 * epsilon_0 * material_factor

    polarization_figure = make_polarization_figure(
        relative_permittivity,
        field_angle_deg,
        field_phase_cycles,
    )
    polarization_plot = mo.as_html(polarization_figure)
    plt.close(polarization_figure)

    mo.vstack(
        [
            mo.hstack(
                [
                    relative_permittivity_slider,
                    field_angle_slider,
                    field_phase_slider,
                ],
                widths="equal",
                wrap=True,
                gap=2,
            ),
            mo.md(
                rf"$K={material_factor:.3f}$"
                rf"$\qquad E_{{\rm in}}/E_i={internal_field_ratio:.3f}$"
                rf"$\qquad P/E_i={polarization_per_incident_field:.3e}\ "
                rf"\mathrm{{C\,V^{{-1}}m^{{-1}}}}$"
            ),
            polarization_plot,
        ],
        gap=1,
    )
    return material_factor, relative_permittivity


@app.cell
def volume_heading(mo):
    mo.md(r"""
    ## Scaling with volume

    $\mathbf{P}$ is a density ($\mathbf{P}$ is the dipole moment, if you're bad at remembering tons of symbols, like me). It tells us how much dipole moment we get per unit volume, and nothing in the last section referred to the size of the drop. A 1 mm drop and a 5 mm drop sitting in the same field have the same $\mathbf{P}$.

    To get the dipole moment of the actual drop we have to integrate over it, and since $\mathbf{P}$ is uniform that integral collapses into a multiplication.

    \[
    \mathbf p=\int_V\mathbf P\,dV=\mathbf PV,
    \qquad
    N_{\rm pairs}\propto V\propto D^3,
    \qquad
    \sum q=0
    \]

    The volume is where size finally enters the problem. Double the diameter and you get eight times the volume, eight times as many stretched pairs, and eight times the dipole moment. The figure below counts the pairs out for you. However much volume we add, the charge still balances.

    Those moments add up cleanly because of the work we did two sections ago. Every pair in the drop feels the same field at the same instant, so all the little moments point the same direction and sum directly. If the drop were big enough for the phase to vary across it, each bit of added volume would buy you less than the last.

    It's worth seeing where this is headed. The scattered field is proportional to $p$, so it goes as $D^3$, and power goes as the field squared. That gives $D^6$, and it's why a few large drops can outshine thousands of small ones. Drag the slider to 2 and watch $\sigma/\sigma_0$ reach 64, which is 18 dB for a single doubling of diameter.
    """)
    return


@app.cell
def volume_control(mo):
    diameter_ratio_slider = mo.ui.slider(
        start=0.5,
        stop=2.0,
        step=0.05,
        value=1.0,
        label=r"$D/D_0$",
        show_value=True,
    )
    return (diameter_ratio_slider,)


@app.cell
def volume_helpers(ACCENT, BACKGROUND, BLUE, Circle, ICE, INK, np, plt):
    def sample_sphere_points(count, radius, seed):
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radial_distance = 0.88 * radius * rng.random(count) ** (1 / 3)
        return directions * radial_distance[:, None]

    def make_volume_figure(diameter_ratio):
        base_count = 20
        scaled_count = max(1, round(base_count * diameter_ratio**3))
        maximum_radius = max(1, diameter_ratio)
        center_offset = 1.05 * (maximum_radius + 1)

        figure, axis = plt.subplots(figsize=(9.0, 4.4))
        figure.patch.set_facecolor(BACKGROUND)
        axis.set_facecolor(BACKGROUND)
        cases = [
            (-center_offset, 1, base_count, 7, r"$D_0$"),
            (
                center_offset,
                diameter_ratio,
                scaled_count,
                7,
                rf"${diameter_ratio:.2f}D_0$",
            ),
        ]

        pair_direction = np.array([0.6, 0.8])
        pair_separation = 0.045
        for center_x, radius, count, seed, label in cases:
            points = sample_sphere_points(count, radius, seed)
            projected = points[:, [0, 2]] + np.array([center_x, 0])
            positive = projected + pair_separation * pair_direction
            negative = projected - pair_separation * pair_direction

            axis.add_patch(
                Circle(
                    (center_x, 0),
                    radius,
                    facecolor=BLUE,
                    edgecolor=INK,
                    linewidth=1.3,
                    alpha=0.16,
                    zorder=0,
                )
            )
            axis.scatter(
                positive[:, 0],
                positive[:, 1],
                s=12,
                color=ACCENT,
                alpha=0.7,
                zorder=2,
            )
            axis.scatter(
                negative[:, 0],
                negative[:, 1],
                s=12,
                color=ICE,
                alpha=0.7,
                zorder=2,
            )
            axis.text(
                center_x,
                -radius - 0.28,
                f"{label}   {count} pairs",
                ha="center",
                va="top",
                color=INK,
                fontsize=11,
            )

        limit = center_offset + maximum_radius + 0.35
        axis.set(
            xlim=(-limit, limit),
            ylim=(-maximum_radius - 0.65, maximum_radius + 0.35),
            xticks=[],
            yticks=[],
            title="representative pairs at fixed volume density",
        )
        axis.title.set_color(INK)
        axis.set_aspect("equal")
        for spine in axis.spines.values():
            spine.set_visible(False)

        figure.tight_layout()
        return figure

    return (make_volume_figure,)


@app.cell
def volume_lab(diameter_ratio_slider, make_volume_figure, mo, plt):
    diameter_ratio = diameter_ratio_slider.value
    relative_volume = diameter_ratio**3
    relative_dipole_moment = relative_volume
    relative_backscatter = diameter_ratio**6

    volume_figure = make_volume_figure(diameter_ratio)
    volume_plot = mo.as_html(volume_figure)
    plt.close(volume_figure)

    mo.vstack(
        [
            diameter_ratio_slider,
            mo.md(
                rf"$D/D_0={diameter_ratio:.2f}$"
                rf"$\qquad p/p_0={relative_dipole_moment:.2f}$"
                rf"$\qquad \sigma/\sigma_0={relative_backscatter:.2f}$"
                r"$\qquad \sum q=0$"
            ),
            volume_plot,
        ],
        gap=1,
    )
    return (diameter_ratio,)


@app.cell
def scattering_heading(mo):
    mo.md(r"""
    ## From dipole moment to backscatter

    We have a dipole and we know how strong it is. Now we need to know where it sends its energy.

    A dipole doesn't radiate the same amount in every direction. It sends the most broadside to itself and nothing at all straight along its own axis. That falloff is $\sin^2\theta$ in power, with $\theta$ measured from $\mathbf{p}$, and it's what the pattern below is showing.

    \[
    \mathbf E_s=
    \frac{k_0^2}{4\pi\varepsilon_0}
    \left[\hat{\mathbf s}\times
    (\mathbf p\times\hat{\mathbf s})\right]
    \frac{e^{-jk_0r}}{r}
    \]

    The cross product is doing all that work. It keeps only the part of $\mathbf{p}$ that looks sideways from where you're standing, so looking down the axis leaves you nothing to see.

    > If you're getting confused with all these directions, scroll down some to see the plot!

    For a radar this works out nicely. The drop gets polarized along the transmitted field, which points across the direction the wave is traveling, so the path back to the antenna sits at $\theta = 90^\circ$ which is the peak.

    \[
    \sigma_{\rm back}
    =\frac{\pi^5}{\lambda^4}|K|^2D^6
    \]

    Evaluating that scattered field at $\theta = 90^\circ$ and substituting $\mathbf{p} = \mathbf{P}V$ gives the backscatter cross section. The $|K|^2$ comes from the material, the $D^6$ from the volume showing up squared, and the $\lambda^{-4}$ from the two factors of $k_0$ in the radiated field.
    """)
    return


@app.cell
def observation_control(mo):
    observation_angle_slider = mo.ui.slider(
        start=0,
        stop=180,
        step=1,
        value=90,
        label=r"$\theta$ from $\mathbf{p}$ ($^\circ$)",
        show_value=True,
    )
    return (observation_angle_slider,)


@app.cell
def scattering_helpers(ACCENT, BACKGROUND, BLUE, INK, MUTED, np, plt):
    def make_scattering_figure(observation_angle_deg):
        figure = plt.figure(figsize=(5.8, 5.2), facecolor=BACKGROUND)
        pattern_axis = figure.add_subplot(projection="polar")
        pattern_axis.set_facecolor(BACKGROUND)

        angles = np.linspace(0, 2 * np.pi, 721)
        field_pattern = np.abs(np.sin(angles))
        power_pattern = np.sin(angles) ** 2
        observation_angle = np.radians(observation_angle_deg)

        pattern_axis.plot(
            angles,
            field_pattern,
            color=BLUE,
            linewidth=2,
            label=r"$|E_s|$",
        )
        pattern_axis.plot(
            angles,
            power_pattern,
            color=ACCENT,
            linewidth=2,
            label="power",
        )
        pattern_axis.scatter(
            [observation_angle],
            [np.sin(observation_angle) ** 2],
            color=ACCENT,
            s=55,
            zorder=5,
        )
        pattern_axis.set_theta_zero_location("N")
        pattern_axis.set_theta_direction(-1)
        pattern_axis.set_rticks([0.5, 1])
        pattern_axis.set_yticklabels([])
        pattern_axis.tick_params(colors=MUTED)
        pattern_axis.grid(color=MUTED, linewidth=0.6, alpha=0.3)
        pattern_axis.spines["polar"].set_color(MUTED)
        pattern_axis.set_title("dipole pattern", color=INK, pad=16)
        pattern_axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.16),
            frameon=False,
            labelcolor=INK,
            ncol=2,
        )
        figure.tight_layout()
        return figure

    return (make_scattering_figure,)


@app.cell
def scattering_lab(
    diameter_mm,
    diameter_ratio,
    make_scattering_figure,
    material_factor,
    mo,
    np,
    observation_angle_slider,
    plt,
    relative_permittivity,
    wavelength_mm,
):
    observation_angle_deg = observation_angle_slider.value
    angular_power_factor = np.sin(np.radians(observation_angle_deg)) ** 2

    scaled_diameter_m = diameter_mm * diameter_ratio * 1e-3
    wavelength_m = wavelength_mm * 1e-3
    monostatic_backscatter_m2 = (
        np.pi**5
        / wavelength_m**4
        * abs(material_factor) ** 2
        * scaled_diameter_m**6
    )
    angle_cross_section_m2 = monostatic_backscatter_m2 * angular_power_factor
    scaled_size_parameter = np.pi * scaled_diameter_m / wavelength_m
    internal_size_parameter = (
        np.sqrt(relative_permittivity) * scaled_size_parameter
    )

    scattering_figure = make_scattering_figure(observation_angle_deg)
    scattering_plot = mo.as_html(scattering_figure)
    plt.close(scattering_figure)

    mo.vstack(
        [
            observation_angle_slider,
            mo.md(
                rf"$\sin^2\theta={angular_power_factor:.3f}$"
                rf"$\qquad \sigma(\theta)={angle_cross_section_m2:.2e}\ "
                r"\mathrm{m^2}$"
                rf"$\qquad x={scaled_size_parameter:.3f}$"
                rf"$\qquad |\sqrt{{\varepsilon_r}}|x={internal_size_parameter:.3f}$"
            ),
            scattering_plot,
        ],
        gap=1,
    )
    return


@app.cell
def dsd_heading(mo):
    mo.md(r"""
    ## Build a drop size distribution

    Everything so far has been one drop. A radar never sees one drop, so we need to know how many of each size are in there. That's the drop size distribution $N(D)$.

    \[
    N(D)=N_w f(\mu)
    \left(\frac{D}{D_0}\right)^\mu
    e^{-(3.67+\mu)D/D_0}
    \]

    This is the normalized gamma form, and the three sliders are its three knobs. $N_w$ scales how many drops there are, $D_0$ sets the middle of the size range, and $\mu$ controls the shape.

    Measuring and modeling real drop size distributions is a whole field on its own, and I'm not going to do it justice here. If you want it done properly, [Radar Meteorology: A First Course](https://tinyurl.com/meteorology-a-first-course) is the gentlest starting point, [Polarimetric Doppler Weather Radar](https://tinyurl.com/polarimetric-chandra) is where this normalized gamma form comes from, and [Doppler Radar and Weather Observations](https://tinyurl.com/doppler-radar-observations) covers the moments and how they connect to rain rate.

    \[
    M_n=\int_0^\infty D^nN(D)\,dD,
    \qquad
    M_0=N_t,
    \qquad
    M_3\propto \text{water volume},
    \qquad
    Z=M_6
    \]

    Those integrals are the moments of the distribution, and each power of $D$ pulls out a different physical quantity. $M_0$ is the number of drops per cubic meter. $M_3$ is proportional to how much water is actually up there. $M_6$ is $Z$, since backscatter from a single drop goes as $D^6$.

    Reflectivity is just the sixth moment of the drop size distribution, and every step we walked through is why the exponent is a 6.

    Watch $Z$ and $M_3$ as you move $D_0$. They pull apart fast, which is why reflectivity alone is a rough way to guess rainfall.
    """)
    return


@app.cell
def dsd_math(gamma, np):
    def normalized_gamma_distribution(
        diameter_mm,
        log10_nw,
        median_volume_diameter_mm,
        shape_parameter,
    ):
        number_concentration = 10**log10_nw
        normalization = (
            6
            / 3.67**4
            * (3.67 + shape_parameter) ** (shape_parameter + 4)
            / gamma(shape_parameter + 4)
        )
        return (
            number_concentration
            * normalization
            * (diameter_mm / median_volume_diameter_mm) ** shape_parameter
            * np.exp(
                -(3.67 + shape_parameter)
                * diameter_mm
                / median_volume_diameter_mm
            )
        )

    def distribution_moment(diameter_mm, distribution, order):
        return np.trapezoid(diameter_mm**order * distribution, diameter_mm)

    def rain_rate(diameter_mm, distribution):
        drop_mass_g = np.pi * diameter_mm**3 / 6000
        fall_speed_mps = 17.67 * (diameter_mm / 10) ** 0.67
        return 3.6 * np.trapezoid(
            drop_mass_g * fall_speed_mps * distribution,
            diameter_mm,
        )

    return distribution_moment, normalized_gamma_distribution, rain_rate


@app.cell
def dsd_controls(mo):
    median_volume_diameter_slider = mo.ui.slider(
        start=0.4,
        stop=3.5,
        step=0.05,
        value=1.5,
        label=r"$D_0$ (mm)",
        show_value=True,
    )
    shape_parameter_slider = mo.ui.slider(
        start=-0.9,
        stop=10.0,
        step=0.1,
        value=2.0,
        label=r"$\mu$",
        show_value=True,
    )
    log10_nw_slider = mo.ui.slider(
        start=2.0,
        stop=5.5,
        step=0.05,
        value=4.0,
        label=r"$\log_{10}N_w$",
        show_value=True,
    )
    return (
        log10_nw_slider,
        median_volume_diameter_slider,
        shape_parameter_slider,
    )


@app.cell
def dsd_plot_helper(
    ACCENT,
    BACKGROUND,
    BLUE,
    GREEN,
    MUTED,
    np,
    plt,
    style_axes,
):
    def make_dsd_weighting_figure(
        diameter_mm,
        distribution,
        median_volume_diameter_mm,
    ):
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(9.2, 7.0),
            sharex=True,
            gridspec_kw={"height_ratios": [1.2, 1, 1]},
        )
        figure.patch.set_facecolor(BACKGROUND)
        number_axis, water_axis, reflectivity_axis = axes
        number_upper = max(1e6, 10 ** np.ceil(np.log10(distribution.max())))

        number_axis.semilogy(
            diameter_mm,
            distribution,
            color=BLUE,
            linewidth=2.4,
        )
        number_axis.set(
            xlim=(0, 8),
            ylim=(1e-2, number_upper),
            ylabel=r"$N(D)$",
            title="the same distribution, weighted three ways",
        )

        for axis, order, color, label in (
            (water_axis, 3, GREEN, r"water: $D^3N(D)$"),
            (reflectivity_axis, 6, ACCENT, r"reflectivity: $D^6N(D)$"),
        ):
            weighted_distribution = diameter_mm**order * distribution
            normalized_weight = weighted_distribution / np.trapezoid(
                weighted_distribution,
                diameter_mm,
            )
            axis.fill_between(
                diameter_mm,
                normalized_weight,
                color=color,
                alpha=0.18,
            )
            axis.plot(
                diameter_mm,
                normalized_weight,
                color=color,
                linewidth=2.4,
            )
            axis.set_ylabel(label)
            axis.set_ylim(0, 1.08 * normalized_weight.max())
            axis.set_yticks([])

        for axis in axes:
            axis.axvline(
                median_volume_diameter_mm,
                color=MUTED,
                linewidth=1,
                linestyle="--",
                alpha=0.65,
            )
            style_axes(axis)
        reflectivity_axis.set_xlabel(r"$D$ (mm)")

        figure.tight_layout(h_pad=1.2)
        return figure

    return (make_dsd_weighting_figure,)


@app.cell
def dsd_builder_lab(
    distribution_moment,
    log10_nw_slider,
    make_dsd_weighting_figure,
    median_volume_diameter_slider,
    mo,
    normalized_gamma_distribution,
    np,
    plt,
    rain_rate,
    shape_parameter_slider,
):
    log10_nw = log10_nw_slider.value
    median_volume_diameter_mm = median_volume_diameter_slider.value
    shape_parameter = shape_parameter_slider.value

    dsd_diameter_mm = np.linspace(0.05, 8.0, 1000)
    custom_dsd = normalized_gamma_distribution(
        dsd_diameter_mm,
        log10_nw,
        median_volume_diameter_mm,
        shape_parameter,
    )
    total_number = distribution_moment(dsd_diameter_mm, custom_dsd, 0)
    water_moment = distribution_moment(dsd_diameter_mm, custom_dsd, 3)
    reflectivity_factor = distribution_moment(
        dsd_diameter_mm,
        custom_dsd,
        6,
    )
    reflectivity_dbz = 10 * np.log10(reflectivity_factor)
    custom_rain_rate = rain_rate(dsd_diameter_mm, custom_dsd)

    dsd_figure = make_dsd_weighting_figure(
        dsd_diameter_mm,
        custom_dsd,
        median_volume_diameter_mm,
    )
    dsd_plot = mo.as_html(dsd_figure)
    plt.close(dsd_figure)

    controls = mo.hstack(
        [
            median_volume_diameter_slider,
            shape_parameter_slider,
            log10_nw_slider,
        ],
        widths="equal",
        wrap=True,
        gap=2,
    )
    mo.vstack(
        [
            controls,
            mo.md(
                rf"$M_0={total_number:.2e}\ \mathrm{{m^{{-3}}}}$"
                rf"$\qquad M_3={water_moment:.2e}\ "
                r"\mathrm{mm^3\,m^{-3}}$"
                rf"$\qquad Z={reflectivity_factor:.2e}\ "
                r"\mathrm{mm^6\,m^{-3}}$"
                rf"$={reflectivity_dbz:.1f}\ \mathrm{{dBZ}}$"
                rf"$\qquad R\approx{custom_rain_rate:.1f}\ \mathrm{{mm\,h^{{-1}}}}$"
            ),
            dsd_plot,
        ],
        gap=1,
    )
    return


@app.cell
def equal_water_heading(mo):
    mo.md(r"""
    ## Same water, different reflectivity
    """)
    return


@app.cell
def equal_water_control(mo):
    small_drop_count_slider = mo.ui.slider(
        start=1,
        stop=64,
        step=1,
        value=8,
        label="number of 1 mm drops",
        show_value=True,
    )
    return (small_drop_count_slider,)


@app.cell
def equal_water_helper(ACCENT, BACKGROUND, BLUE, INK, np, plt, style_axes):
    def make_equal_water_figure(small_drop_count):
        equivalent_diameter = small_drop_count ** (1 / 3)
        grid_width = int(np.ceil(np.sqrt(small_drop_count)))
        rows = int(np.ceil(small_drop_count / grid_width))

        figure, (drops_axis, moment_axis) = plt.subplots(
            1,
            2,
            figsize=(9.6, 4.2),
            gridspec_kw={"width_ratios": [1.35, 1]},
        )
        figure.patch.set_facecolor(BACKGROUND)
        drops_axis.set_facecolor(BACKGROUND)

        x_positions = np.arange(small_drop_count) % grid_width
        y_positions = np.arange(small_drop_count) // grid_width
        x_positions = x_positions - (grid_width - 1) / 2
        y_positions = y_positions - (rows - 1) / 2
        spacing = 0.30
        x_positions = -1.7 + spacing * x_positions
        y_positions = spacing * y_positions
        drops_axis.scatter(
            x_positions,
            y_positions,
            s=95,
            color=BLUE,
            edgecolor=BACKGROUND,
            linewidth=0.5,
            alpha=0.8,
        )
        drops_axis.scatter(
            [1.55],
            [0],
            s=95 * equivalent_diameter**2,
            color=ACCENT,
            edgecolor=BACKGROUND,
            linewidth=0.8,
            alpha=0.9,
        )
        drops_axis.text(
            -1.7,
            -1.45,
            rf"{small_drop_count} $\times$ 1 mm",
            ha="center",
            color=INK,
            fontsize=11,
        )
        drops_axis.text(
            1.55,
            -1.45,
            rf"1 $\times$ {equivalent_diameter:.2f} mm",
            ha="center",
            color=INK,
            fontsize=11,
        )
        drops_axis.set(
            xlim=(-3.1, 2.7),
            ylim=(-1.7, 1.7),
            xticks=[],
            yticks=[],
            title="equal liquid-water volume",
        )
        drops_axis.title.set_color(INK)
        drops_axis.set_aspect("equal")
        for spine in drops_axis.spines.values():
            spine.set_visible(False)

        moment_three = [small_drop_count, equivalent_diameter**3]
        moment_six = [small_drop_count, equivalent_diameter**6]
        x_locations = np.array([0, 1])
        bar_width = 0.34
        moment_axis.bar(
            x_locations - bar_width / 2,
            moment_three,
            width=bar_width,
            color=BLUE,
            label=r"$\sum D^3$",
        )
        moment_axis.bar(
            x_locations + bar_width / 2,
            moment_six,
            width=bar_width,
            color=ACCENT,
            label=r"$\sum D^6$",
        )
        moment_axis.set(
            xticks=x_locations,
            xticklabels=["small drops", "one large drop"],
            ylabel="relative sum",
            title="the sixth moment changes",
        )
        moment_axis.legend(frameon=False, labelcolor=INK)
        style_axes(moment_axis, grid=False)

        figure.tight_layout(w_pad=3)
        return figure, equivalent_diameter

    return (make_equal_water_figure,)


@app.cell
def equal_water_lab(
    make_equal_water_figure,
    mo,
    np,
    plt,
    small_drop_count_slider,
):
    small_drop_count = small_drop_count_slider.value
    equal_water_figure, equivalent_large_diameter_mm = make_equal_water_figure(
        small_drop_count
    )
    equal_water_plot = mo.as_html(equal_water_figure)
    plt.close(equal_water_figure)

    small_drop_reflectivity_sum = small_drop_count
    large_drop_reflectivity_sum = equivalent_large_diameter_mm**6
    reflectivity_ratio = (
        large_drop_reflectivity_sum / small_drop_reflectivity_sum
    )
    reflectivity_difference_db = 10 * np.log10(reflectivity_ratio)

    mo.vstack(
        [
            small_drop_count_slider,
            mo.md(
                rf"$D_{{\rm large}}={equivalent_large_diameter_mm:.2f}\ "
                rf"\mathrm{{mm}}$"
                rf"$\qquad Z_{{\rm small}}={small_drop_reflectivity_sum:.1f}$"
                rf"$\qquad Z_{{\rm large}}={large_drop_reflectivity_sum:.1f}$"
                rf"$\qquad {reflectivity_ratio:.1f}\times"
                rf"={reflectivity_difference_db:.1f}\ \mathrm{{dB}}$"
            ),
            equal_water_plot,
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
