#%%
import pandas as pd
import matplotlib.pyplot as plt
print(pd.__version__)
#%%
data = pd.read_csv('Misura_flusso_ripulito.csv', encoding='utf-8')
#print(data.head)
#print(data["Misure 1"])
print(data.columns.tolist())

#Rinominiamo le colonne di interesse
data.columns = [col.strip() for col in data.columns]
print(data.columns)

col_angolo = "Angel []"
col_flux = "Flux [Hz]"
col_errore = "ERR..1"
print(data["Angel []"].head(10))  # o df[col_angolo]
print(data["Flux [Hz]"].head(10))  # o df[col_flux]
print(data["ERR..1"].head(10))  # o df[col_errore]

# Conversione dei valori con la virgola
data[col_flux] = data[col_flux].str.replace(",", ".").astype(float)
data[col_errore] = data[col_errore].str.replace(",", ".").astype(float)
# %%
plt.figure(figsize=(10,6))
print(data[[col_angolo, col_flux, col_errore]].dropna())
print(f"Numero di righe valide: {len(data.dropna(subset=[col_angolo, col_flux, col_errore]))}")

plt.errorbar(data[col_angolo],data[col_flux],yerr = data[col_errore], fmt= 'o',capsize = 5,markersize = 5)
plt.xlabel("Angolo")
plt.ylabel("Flux [Hz]")
plt.title("Flusso in funzione dell'angolo")
plt.grid(True)
plt.show()
# %%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Estrazione dati
x = data[col_angolo].values
y = data[col_flux].values
y_err = data[col_errore].values

# Definizione della funzione gaussiana
def gaussiana(x, A, mu, sigma):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2))

# Stima iniziale dei parametri (aiuta il fit a convergere!)
A0 = np.max(y)
mu0 = x[np.argmax(y)]
sigma0 = 20  # stima iniziale arbitraria

p0 = [A0, mu0, sigma0]

valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(y_err)) & (~np.isinf(y_err))

x_clean = x[valid_mask]
y_clean = y[valid_mask]
y_err_clean = y_err[valid_mask]

popt, pcov = curve_fit(gaussiana, x_clean, y_clean, sigma=y_err_clean, p0=p0, absolute_sigma=True)

# Fit con errore sui dati y
#popt, pcov = curve_fit(gaussiana, x, y, sigma=y_err, p0=p0, absolute_sigma=True)

# Estrazione dei parametri e incertezze
A_fit, mu_fit, sigma_fit = popt
dA, dmu, dsigma = np.sqrt(np.diag(pcov))

print(f"Fit gaussiano trovato:")
print(f"A     = {A_fit:.3f} ± {dA:.3f}")
print(f"mu    = {mu_fit:.3f} ± {dmu:.3f}")
print(f"sigma = {sigma_fit:.3f} ± {dsigma:.3f}")

# Valori per il fit da plottare
x_fit = np.linspace(min(x), max(x), 500)
y_fit = gaussiana(x_fit, *popt)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.errorbar(x, y, yerr=y_err, fmt='o', label='Dati', capsize=5)
plt.plot(x_fit, y_fit, label='Fit gaussiano', color='red')
plt.xlabel("Angolo [°]")
plt.ylabel("Flusso [Hz]")
plt.title("Fit Gaussiano del Flusso")
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  # opzionale
plt.tight_layout()
plt.show()

# %%
