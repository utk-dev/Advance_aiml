import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.widgets import Slider

# Use a safe style fallback
try:
    plt.style.use('./deeplearning.mplstyle')
except:
    plt.style.use('ggplot')

try:
    from lab_utils_common import dlc
except ImportError:
    dlc = {"dldarkred": "#C00000"}  # fallback color

def plt_softmax(softmax_fn):
    """
    Visualize softmax with interactive sliders.
    
    Args:
        softmax_fn: A callable that accepts a numpy array and returns softmax probabilities.
                    Example: lambda z: np.exp(z) / np.sum(np.exp(z))
    """
    # --- Guard: ensure softmax_fn is actually callable ---
    if not callable(softmax_fn):
        raise TypeError(
            f"Expected a callable softmax function, got {type(softmax_fn).__name__}. "
            "Usage: plt_softmax(my_softmax_function)"
        )

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(bottom=0.35)

    # Slider axes: [left, bottom, width, height]
    axz0 = fig.add_axes([0.15, 0.10, 0.30, 0.03])
    axz1 = fig.add_axes([0.15, 0.15, 0.30, 0.03])
    axz2 = fig.add_axes([0.15, 0.20, 0.30, 0.03])
    axz3 = fig.add_axes([0.15, 0.25, 0.30, 0.03])

    z0 = Slider(axz0, 'z0', 0.1, 10.0, valinit=1, valstep=0.1)
    z1 = Slider(axz1, 'z1', 0.1, 10.0, valinit=2, valstep=0.1)
    z2 = Slider(axz2, 'z2', 0.1, 10.0, valinit=3, valstep=0.1)
    z3 = Slider(axz3, 'z3', 0.1, 10.0, valinit=4, valstep=0.1)

    # Initial values
    z_init = np.array([z0.val, z1.val, z2.val, z3.val])
    a_init = softmax_fn(z_init)

    z_labels = np.array(['z0', 'z1', 'z2', 'z3'])
    a_labels = np.array(['a0', 'a1', 'a2', 'a3'])

    # Left bar chart: z inputs
    bar  = ax[0].barh(z_labels, z_init, height=0.6, align='center')
    bars = bar.get_children()
    ax[0].set_xlim([0, 10])
    ax[0].set_title("z input to softmax")

    # Right bar chart: softmax outputs
    sbar  = ax[1].barh(a_labels, a_init, height=0.6, align='center',
                       color=dlc["dldarkred"])
    sbars = sbar.get_children()
    ax[1].set_xlim([0, 1])
    ax[1].set_title("softmax(z)")

    def update(val):
        z_vals = np.array([z0.val, z1.val, z2.val, z3.val])
        a_vals = softmax_fn(z_vals)

        for i, b in enumerate(bars):
            b.set_width(z_vals[i])
        for i, b in enumerate(sbars):
            b.set_width(a_vals[i])

        fig.canvas.draw_idle()

    z0.on_changed(update)
    z1.on_changed(update)
    z2.on_changed(update)
    z3.on_changed(update)

    plt.show()