import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 500)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)

hline = ax.axhline(color="k", ls="--", lw=0.8)
vline = ax.axvline(color="k", ls="--", lw=0.8)
texto = ax.text(0.02, 0.95, "", transform=ax.transAxes)


def mover(event):
    if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
        x0 = event.xdata
        y0 = event.ydata

        hline.set_ydata([y0])
        vline.set_xdata([x0])
        texto.set_text(f"x = {x0:.3f}\ny = {y0:.3f}")
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", mover)

plt.show()
