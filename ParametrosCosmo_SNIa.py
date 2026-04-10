import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize, fsolve


# Definimos la función distancia por luminosidad, con parametros de entrada un vector con distintos z, w_0, w_a, Omega_M.
def H0_dl(z, w_0, w_a, Omega_M=0.334):
    Omega_DE = 1 - Omega_M

    def f(z):
        return 1 / np.sqrt(
            Omega_M * (1 + z) ** 3
            + Omega_DE
            * (1 + z) ** (3 * (1 + w_0 + w_a))
            * np.exp(-3 * w_a * (z / (1 + z)))
        )

    integral = quad(f, 0, z)[0]
    return (1 + z) * integral


def H0_dl_vec(z, w_0, w_a, Omega_M=0.334):
    H0_dl_vectorizada = np.vectorize(H0_dl)
    return H0_dl_vectorizada(z, w_0, w_a, Omega_M)


# Dibujamos la distancia por luminosidad para w_0=-1 y w_a=-5,0,5 en el rango z=(0,1)
z = np.linspace(0, 1, 100)
w_a = [-5, 0, 5]
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.grid(True)

ax.plot(z, H0_dl_vec(z, -1, w_a[0]), "-", color="red", label=r"$w_a=-5$")
ax.plot(z, H0_dl_vec(z, -1, w_a[1]), "-", color="blue", label=r"$w_a=0$")
ax.plot(z, H0_dl_vec(z, -1, w_a[2]), "-", color="orange", label=r"$w_a=5$")

ax.legend(loc=2, fontsize=16)
x_min, x_max, y_min, y_max = ax.axis("tight")
ax.axis([x_min, x_max, y_min, y_max])

ax.axes.xaxis.set_label_text("Redshift " + r"$z$")
ax.axes.yaxis.set_label_text(r"$(H_0/c)\, d_L$")

fig.suptitle(
    "Distancia por Luminosidad en modelo " + r"$w_0=-1$", fontsize=20, fontweight="bold"
)
# plt.show()
fig.savefig("distancia_luminosidad.pdf")
fig.savefig("distancia_luminosidad.png")
plt.close(fig)

# Calculamos el valor para z=0.2, w_0=-1 y w_a=5
print("El valor de H_0 d_L es: ", H0_dl(0.2, -1, 5))


# Construimos la funcion modulo de distancias
def mu_th(z, h, w_0, w_a, Omega_M=0.334):
    H0_1 = 2997.9 / h
    return 5 * np.log10(H0_dl_vec(z, w_0, w_a, Omega_M) * H0_1) + 25


def mu_th_vec(z, h, w_0, w_a, Omega_M=0.334):
    mu_th_vectorizada = np.vectorize(mu_th)
    return mu_th_vectorizada(z, h, w_0, w_a, Omega_M)


# Calculamos el valor para z=0.2, h=0.7 w_0=-1 y w_a=5
print("El valor de mu_th es: ", mu_th(0.2, 0.7, -1, 5))

# Cargamos los datos PantheonSHOES con su matriz de covarianza
zobs, muobs = np.loadtxt("PantheonSH0ES_unique_data.txt", skiprows=1, unpack=True)
cov = np.loadtxt("PantheonSH0ES_unique_cov.gz")


# Definimos nuestro likelihood mediante la función chi-cuadrado asociada.
cov_inv = np.linalg.inv(cov)


def chi_cuad(h, w_0, w_a, Omega_M=0.334):
    # Definimos el vector de diferencias
    delta = muobs - mu_th_vec(z=zobs, h=h, w_0=w_0, w_a=w_a, Omega_M=Omega_M)
    # Devolvemos chi cuadrado
    return np.dot(np.transpose(delta), np.dot(cov_inv, delta))


def chi_cuad_vec(h, w_0, w_a, Omega_M=0.334):
    chi_cuad_vectorizada = np.vectorize(chi_cuad)
    return chi_cuad_vectorizada(h, w_0, w_a, Omega_M=0.334)


# Calculamos el valor para h=0.65, w_0=-1 y w_a=5.
print("El valor de chi_cuadrado es: ", chi_cuad_vec(0.65, -1, 5))

# AJUSTE PARA LA ECUACIÓN DE ESTADO PARA LA ENERGÍA OSCURA

# Obtención de w_0 para un modelo con h, w_a y Omega_M fijos (Modelo wCDM).
H_0 = 73.04
w_a = 0
Omega_M = 0.334


# Minimizamos chi_cuad para ajustar nuestros datos con el likelihood
def chi_cuad_min_w_0(w_0):
    return chi_cuad_vec(H_0 / 100, w_0, w_a, Omega_M)


# Representamos la función para identificar el mínimo
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.grid(True)

w = np.linspace(-3, 1, 100)
chi2_vals = chi_cuad_min_w_0(w)
ax.plot(w, chi2_vals, "-", color="red")

x_min, x_max, y_min, y_max = ax.axis("tight")
ax.axis([x_min, x_max, y_min, y_max])

ax.axes.xaxis.set_label_text("Parámetro " + r"$\omega_0$")
ax.axes.yaxis.set_label_text("Estimador estadístico " + r"$\chi^2$")

fig.suptitle(
    "Parámetro " + r"$\omega_0$" + " en modelo wCDM", fontsize=20, fontweight="bold"
)
# plt.show()
plt.close(fig)

# Ahora, minimizamos la función con el modulo scipy.optimize, utilizando como punto inicial w_0=-1
minimo_1 = minimize(chi_cuad_min_w_0, x0=-1)
w_min = minimo_1.x[0]
chi_cuad_minimo = chi_cuad_min_w_0(w_min)
print(
    f"Mínimo para wCDM encontrado en:\n   w_0 = {w_min:.4f} con chi^2 = {chi_cuad_minimo:.4f}"
)

# Calculamos los errores a una sigma por la derecha y por la izquierda
# Definimos los Delta chi-cuadrado para cada sigma teniendo en cuenta que tenemos un único parámetro, es decir, sigma_n->Delta chi-cuadrado=n^2
delta_chi_1 = np.arange(1, 4) ** 2
confianza = [68.27, 95.45, 99.73]
colors = ["red", "blue", "orange"]

# Calculamos los intervalos a distintos sigmas
w_l = [0, 0, 0]
w_r = [0, 0, 0]
for i in range(3):
    delta_chi = delta_chi_1[i]
    probabilidad = confianza[i]
    f = lambda w_0: chi_cuad_min_w_0(w_0) - chi_cuad_minimo - delta_chi
    w_left = fsolve(f, x0=w_min - 0.5)[0]
    w_right = fsolve(f, x0=w_min + 0.5)[0]
    print(
        f"Intervalo para sigma {i + 1} ({probabilidad} %): w en [{w_left:.4f}, {w_right:.4f}]"
    )
    w_l[i] = w_left
    w_r[i] = w_right

# Visualizamos los distintos intervalos
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.grid(True)

w = np.linspace(-1.5, -0.5, 100)
chi2_vals = chi_cuad_min_w_0(w)
ax.plot(w, chi2_vals, "-", color="red")
plt.axvline(w_min, color="black", label=r"$\omega_0$ ajustado")
for i in range(3):
    plt.axvline(
        x=w_l[i],
        linestyle="dashed",
        color=colors[i],
        label=f"$\\sigma_{i + 1}$",
    )
    plt.axvline(x=w_r[i], linestyle="dashed", color=colors[i])
ax.legend(loc=2, fontsize=16)
x_min, x_max, y_min, y_max = ax.axis("tight")
ax.axis([x_min, x_max, y_min, y_max])

ax.axes.xaxis.set_label_text("Parámetro " + r"$\omega_0$")
ax.axes.yaxis.set_label_text("Estimador estadístico " + r"$\chi^2$")

fig.suptitle(
    "Parámetro " + r"$\omega_0$" + " en modelo wCDM", fontsize=20, fontweight="bold"
)
# plt.show()
fig.savefig("w_0_wCMD.pdf")
fig.savefig("w_0_wCMD.png")
plt.close(fig)


# Comparamos nuestros resultados con el valor obtenido por Planck en el 2018, al que consideramos como valor teórico (w_0_Planck=-1.028 ± 0.032). Observando una clara compatibilidad con nuestros resultados, ya que para todos los sigmas los intervalos de incertidumbre se solapan, podemos estudiar el error o desviación cometido.
def error(theo, exp):
    return (abs(exp - theo)) * 100 / abs(theo)


w_0_Planck = -1.028
print(
    f"El error relativo cometido en nuestro análisis es del {error(theo=w_0_Planck, exp=w_min):.4f}%."
)

# Representamos el modulo de la distancia en función de z para el valor ajustado de w_0 y el valor teórico de Planck, junto con los datos observados.
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.grid(True)

z = np.linspace(min(zobs), max(zobs), 100)
modulo_ajustado = mu_th_vec(z, H_0 / 100, w_min, w_a, Omega_M)
modulo_Planck = mu_th_vec(z, H_0 / 100, w_0_Planck, w_a, Omega_M)

# Error para datos experimentales
sigmaobs = np.sqrt(np.diag(v=cov))

ax.errorbar(
    zobs, muobs, yerr=sigmaobs, fmt="o", color="black", label="Datos experimentales"
)
ax.plot(z, modulo_ajustado, "-", color="red", label=r"Modelo teórico ajustado")
ax.plot(z, modulo_Planck, "-", color="blue", label=r"Modelo con Planck")

ax.legend(loc=2, fontsize=16)
x_min, x_max, y_min, y_max = ax.axis("tight")
ax.axis([x_min, x_max, y_min, y_max])
ax.axes.xaxis.set_label_text("Redshift " + r"$z$")
ax.axes.yaxis.set_label_text("Módulo de distancia " + r"$\mu$")
fig.suptitle(
    "Modulo de distancia en función de el redshift " + r"$\mu(z)$",
    fontsize=20,
    fontweight="bold",
)
# plt.show()
fig.savefig("Modelos_wCMD.pdf")
fig.savefig("Modelos_wCMD.png")
plt.close(fig)

# Utilizando el valor que mejor ajusta nuestros datos para w_0 calculamos la distancia por luminosidad y la distancia comovil a una supernova con z=0.4.


# Definimos el factor de escala
def a(z):
    return 1 / (1 + z)


def a_vec(z):
    a_vectorizada = np.vectorize(a)
    return a_vectorizada(z)


h = H_0 / 100
d_L = H0_dl(0.4, w_min, w_a, Omega_M) * (2997.9 / h)
d_C = d_L * a(0.4)
print(f"Distancia por luminosidad: {d_L:.4f} Mpc.")
print(f"Distancia comóvil: {d_C:.4f} Mpc.")

# Obtención de w_0 y h para un modelo con w_a y Omega_M fijos (Modelo wCDM).
w_a = 0
Omega_M = 0.334


# Minimizamos chi_cuad para ajustar nuestros datos con el likelihood, utilizando como puntos iniciales la h de teórica y el w mínimo del apartado anterior.
def chi_cuad_min_w_0_h(h, w_0):
    return chi_cuad_vec(h, w_0, w_a, Omega_M)


# Añadimos limites a minimize para que no salga de valores físicos y no tome valores negativos en el logaritmo.
minimo_2 = minimize(
    lambda x: chi_cuad_min_w_0_h(x[0], x[1]),
    x0=[h, w_min],
    bounds=[(0.5, 1.0), (-3.0, 0.0)],
)
h_min_1 = minimo_2.x[0]
w_min_2 = minimo_2.x[1]
chi_cuad_minimo_2 = chi_cuad_min_w_0_h(h_min_1, w_min_2)
print(
    f"Mínimo para wCDM encontrado en: \n   (h, w) = ({h_min_1:.4f}, {w_min_2:.4f}) con chi^2 = {chi_cuad_minimo_2:.4f}"
)

# Representamos en un gráfico los contornos a uno, dos y tres sigmas junto al valor de mejor ajuste, y los resultados del apartado anterior.
h_grid = np.linspace(h_min_1 - 0.05, h_min_1 + 0.015, 50)
w0_grid = np.linspace(w_min_2 - 0.5, w_min_2 + 0.5, 50)
hh, ww = np.meshgrid(h_grid, w0_grid)
chi2_grid = chi_cuad_min_w_0_h(hh, ww)

delta_chi2_2d = [2.30, 6.18, 11.83]

# Niveles para las líneas de contorno
levels = [chi_cuad_minimo_2 + d for d in delta_chi2_2d]

# Niveles para el relleno: hay que incluir el mínimo
levels_fill = [chi_cuad_minimo_2] + levels

h_Planck = 0.6736

fig, ax = plt.subplots(1, figsize=(10, 8))
ax.grid(True, alpha=0.3)

# Relleno de las regiones de confianza
ax.contourf(
    hh,
    ww,
    chi2_grid,
    levels=levels_fill,
    colors=["#FB8B9A", "#FFAE79", "#99C2F1"],
    alpha=0.85,
)

# Líneas de contorno
cont = ax.contour(
    hh,
    ww,
    chi2_grid,
    levels=levels,
    colors=["#FF0022", "#FF6F00", "#003677"],
    linestyles="dashed",
    linewidths=1.8,
)

# Mejor ajuste del modelo 2D (h, w_0) libres
ax.scatter(
    h_min_1,
    w_min_2,
    color="#222222",
    s=90,
    marker="x",
    linewidths=2.2,
    label="Mejor ajuste (h, w_0)",
)

# Mejor ajuste del primer modelo (h fijo = SH0ES)
h = H_0 / 100.0
ax.scatter(
    h,
    w_min,
    color="#E6142C",
    s=85,
    marker="x",
    linewidths=2.2,
    label="Mejor ajuste modelo 1 (h=SH0ES)",
)

# Punto Planck para w_0 con h fijo a SH0ES
ax.scatter(
    h,
    w_0_Planck,
    color="#2F00FF",
    s=85,
    marker="x",
    linewidths=2.2,
    label="w_0 Planck, h=SH0ES",
)

# Punto Planck para w_0 con h fijo a Planck 2018
ax.scatter(
    h_Planck,
    w_0_Planck,
    color="#FF5900",
    s=85,
    marker="x",
    linewidths=2.2,
    label="w_0 Planck, h=Planck",
)

fmt = {levels[0]: r"$1\sigma$", levels[1]: r"$2\sigma$", levels[2]: r"$3\sigma$"}
ax.clabel(cont, inline=True, fontsize=11, fmt=fmt)

ax.set_xlabel(r"$h$", fontsize=13)
ax.set_ylabel(r"$w_0$", fontsize=13)
ax.set_title(r"$\chi^2$ con contornos de confianza en el plano $(h,w_0)$", fontsize=14)
ax.legend(loc="best", fontsize=11, frameon=True)

fig.savefig("Contornos_h_w0_wCDM.pdf")
fig.savefig("Contornos_h_w0_wCDM.png")

# Definimos un puntero para mostrar los valores de h y w_0 al mover el ratón por el gráfico.
hline = ax.axhline(y=w_min_2, color="k", ls="--", lw=0.8)
vline = ax.axvline(x=h_min_1, color="k", ls="--", lw=0.8)
texto = ax.text(
    0.02,
    0.95,
    f"h = {h_min_1:.3f}\nw_0 = {w_min_2:.3f}",
    transform=ax.transAxes,
    va="top",
)


def mover(event):
    if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
        x0 = event.xdata
        y0 = event.ydata

        hline.set_ydata([y0])
        vline.set_xdata([x0])
        texto.set_text(f"h = {x0:.3f}\nw_0 = {y0:.3f}")
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", mover)

plt.show()
