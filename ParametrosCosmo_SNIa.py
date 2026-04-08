import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad


# Definimos las función distancia por luminosidad, con parametros de entrada un vector con distintos z, w_0, w_a, Omega_M.
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


# Dibujamos la distnacia por luminosidad para w_0=-1 y w_a=-5,0,5 en el rango z=(0,1)
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

ax.axes.xaxis.set_label_text("Redshift [km]")
ax.axes.yaxis.set_label_text("Distancia por luminosidad [" r"$H_0$" "Km]")

fig.suptitle(
    "Distancia por Luminosidad en modelo " + r"$w_0=-1$", fontsize=20, fontweight="bold"
)
plt.show()

# Calculamos el valor para z=0.2, w_0=-1 y w_a=5
print("El valor de H_0 d_L es: ", H0_dl(0.2, -1, 5))


# Construimos la funcion modulo de distancias
def mu_th(z, h, w_0, w_a, Omega_M=0.334):
    H0_1 = 2997.9 / h
    return 5 * np.log10(H0_dl_vec(z, w_0, w_a, Omega_M) * H0_1) + 25


# Calculamos el valor para z=0.2, h=0.7 w_0=-1 y w_a=5
print("El valor de mu_th es: ", mu_th(0.2, 0.7, -1, 5))
