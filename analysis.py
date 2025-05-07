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
"""
# Definizione della funzione gaussiana
def gaussiana(x, A, mu, sigma):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2))
"""
def cos2_fit(theta, A, mu, C):
    return A * np.cos(np.radians(theta - mu))**2 + C
# Stima iniziale dei parametri (aiuta il fit a convergere!)


# valori iniziali ragionevoli per il fit

valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(y_err)) & (~np.isinf(y_err))
x_clean = x[valid_mask]
y_clean = y[valid_mask]
y_err_clean = y_err[valid_mask]

A0 = max(y_clean) - min(y_clean)
mu0 = 0  # ipotizziamo un massimo vicino a 0
C0 = min(y_clean)
p0 = [A0, mu0, C0]
# esegui il fit
popt, pcov = curve_fit(cos2_fit, x_clean, y_clean, sigma=y_err_clean, p0=p0, absolute_sigma=True)

# parametri del fit
A_fit, mu_fit, C_fit = popt
A_err, mu_err, C_err = np.sqrt(np.diag(pcov))

print(f"A     = {A_fit:.3f} ± {A_err:.3f}")
print(f"mu    = {mu_fit:.3f} ± {mu_err:.3f}")
print(f"C     = {C_fit:.3f} ± {C_err:.3f}")

# visualizzazione del fit
theta_fit = np.linspace(-90, 90, 400)  
flux_fit = cos2_fit(theta_fit, A_fit, mu_fit, C_fit)

plt.figure(figsize=(10, 6))
plt.errorbar(x_clean, y_clean, yerr=y_err_clean, fmt='o', label='Dati')
plt.plot(theta_fit, flux_fit, color='red', label='Fit cos²')
plt.xlabel('Angolo [gradi]')
plt.ylabel('Flusso [Hz]')
plt.title('Fit del flusso dei raggi cosmici con funzione cos²')
plt.legend()
plt.grid()
plt.show()

# %%
