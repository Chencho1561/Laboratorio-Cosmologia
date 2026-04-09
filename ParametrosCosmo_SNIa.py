import numpy as np
import matplotlib as mpl
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
fig, ax = plt.subplots(1, figsize=(15, 10))
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
plt.show()
fig.savefig("w_0_wCMD.pdf")
fig.savefig("w_0_wCMD.png")
# plt.close(fig)


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
    "Modulo de distancia en función de el redshift" + r"$\mu(z)$",
    fontsize=20,
    fontweight="bold",
)
plt.show()
fig.savefig("Modelos_wCMD.pdf")
fig.savefig("Modelos_wCMD.png")
# plt.close(fig)

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
