# /// script
# dependencies = ["marimo", "matplotlib", "numpy"]
# [tool.marimo.display]
# theme = "light"
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use("default")


    return np, plt


@app.cell
def _(np, plt):
    x = np.linspace(-65, 1465, 6001)
    tau = (x - 700) / 700 * 1.08
    envelope = np.clip(1 - np.abs(tau), 0, None)
    doppler = np.linspace(-0.87 * 25, 0.87 * 25, 67)
    scale = (doppler[1] - doppler[0]) / 14.8
    horizon = np.full_like(x, np.inf)

    fig, ax = plt.subplots(figsize=(9, 5.25), facecolor="white")
    for row in range(66, -1, -1):
        response = envelope * np.abs(np.sinc((doppler[row] + 25 * tau) * envelope))
        y = doppler[row] - 125 * scale * response
        ax.plot(tau, np.where(y < horizon - 0.12 * scale, y, np.nan), color="black", linewidth=0.6)
        horizon = np.minimum(horizon, y)

    ax.set(
        xlim=(-1.058, 1.058),
        ylim=(doppler[0] + 810 * scale, doppler[0] + 10 * scale),
        title="LFM ambiguity ridgelines",
        xlabel=r"Normalized delay ($\tau / T$)",
        ylabel=r"Normalized Doppler ($vT$), stacked cuts",
    )
    fig.tight_layout()
    plt.close(fig)
    fig
    return


if __name__ == "__main__":
    app.run()
