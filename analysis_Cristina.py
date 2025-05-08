#%%
# --- Step 1: Ver. ---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
print(pd.__version__)
#%%
# --- Step 2: Spezza le righe in base alla virgola ---
data = pd.read_csv('Misura_flusso_ripulito.csv', encoding='utf-8')

# --- Visualizza i primi 10 dati per verificare ---
col_angolo = "Angel []"
col_flux = "Flux [Hz]"
col_errore = "ERR."
print(data.head(10))
print(data[col_angolo].head(10))  # o df[col_angolo]
print(data[col_flux].head(10))  # o df[col_flux]
print(data[col_errore].head(10))  # o df[col_errore]
# Conversione dei valori con la virgola
data[col_flux] = data[col_flux].str.replace(",", ".").astype(float)
data[col_errore] = data[col_errore].str.replace(",", ".").astype(float)

#%%
# --- Step 3:
plt.figure(figsize=(10,6))
print(data[[col_angolo, col_flux, col_errore]].dropna())
print(f"Numero di righe valide: {len(data.dropna(subset=[col_angolo, col_flux, col_errore]))}")
plt.errorbar(data[col_angolo],data[col_flux],yerr = data[col_errore], fmt= 'o',capsize = 5,markersize = 5)
plt.xlabel("Angolo")
plt.ylabel("Flux [Hz]")
plt.title("Flusso in funzione dell'angolo")
plt.grid(True)
plt.show()
#%%
# --- Step 4: FIT GAUSSIANO -----
# FIT GAUSSIANO
from scipy.stats import chi2
# Estrazione dati
x = data[col_angolo].values
y = data[col_flux].values
y_err = data[col_errore].values

# Definizione della funzione gaussiana
def gaussiana(x, A, mu, sigma):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2))
valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(y_err)) & (~np.isinf(y_err)) & (~np.isinf(x)) & (~np.isinf(y))

x_clean = x[valid_mask]
y_clean = y[valid_mask]
y_err_clean = y_err[valid_mask]
# Stima iniziale dei parametri (aiuta il fit a convergere!)
A0 = np.max(y_clean)-np.min(y_clean)
mu0 = x_clean[np.argmax(y_clean)]
sigma0 = (np.max(x_clean)-np.min(x_clean))/4  # stima iniziale arbitraria

p0 = [A0, mu0, sigma0]


# Esegui il fit
popt, pcov = curve_fit(gaussiana, x_clean, y_clean, sigma=y_err_clean, p0=p0, absolute_sigma=True)

# Estrazione dei parametri e incertezze
A_fit, mu_fit, sigma_fit = popt
dA, dmu, dsigma = np.sqrt(np.diag(pcov))

print(f"Fit gaussiano trovato:")
print(f"A     = {A_fit:.3f} ± {dA:.3f}")
print(f"mu    = {mu_fit:.3f} ± {dmu:.3f}")
print(f"sigma = {sigma_fit:.3f} ± {dsigma:.3f}")

# Valori per il fit da plottare
x_fit = np.linspace(min(x_clean), max(x_clean), 500)
y_fit = gaussiana(x_fit, *popt)
# Calcolo del chi quadrato
#y_fit = gaussiana(x_fit, *popt)
print("Lunghezza parametri : ", len(y_clean),len(y_fit),len(y_err_clean))
#%%
Y_pred = gaussiana(x_clean, A_fit,mu_fit,sigma_fit)
print("NaN is y_clean: ", np.isnan(y_clean).any())
print("inf is y_clean: ", np.isinf(y_clean).any())
print("NaN is y_err_clean: ", np.isnan(y_err_clean).any())
print("inf is y_err_clean: ", np.isinf(y_err_clean).any())
print("NaN is y_pred: ", np.isnan(Y_pred).any())
print("inf is y_pred: ", np.isinf(Y_pred).any())

chi2_val = np.sum(((y_clean - Y_pred) ** 2) / (y_err_clean ** 2))
dof = len(y_clean) - len(popt)  # gradi di libertà: numero di dati meno numero di parametri del modello
#%%
# Calcolo del valore p dal chi quadrato
p_val = 1 - chi2.cdf(chi2_val, dof)

# Stampa del chi quadrato e valore p
print(f"Chi quadrato = {chi2_val:.2f}")
print(f"Gradi di libertà = {dof}")
print(f"Valore di p = {p_val:.3f}")

# Valori per il fit da plottare
#x_fit = np.linspace(min(x), max(x), 500)
#y_fit = gaussiana(x_fit, *popt)
#%%
# --- Plot ---
plt.figure(figsize=(10, 6))
plt.errorbar(x_clean, y_clean, yerr=y_err_clean, fmt='o', label='Dati', capsize=5)
plt.plot(x_fit, y_fit, label='Fit gaussiano', color='red')
plt.xlabel("Angolo [°]")
plt.ylabel("Flusso [Hz]")
plt.title("Fit Gaussiano del Flusso")
plt.legend()
plt.grid(True)
#plt.gca().invert_xaxis()  # opzionale
plt.tight_layout()
plt.show()
#%%
# --- Step 4: FIT COS^2 -----
# Estrazione dati
x = data["Angel []"].values
y = data["Flux [Hz]"].values
y_err = data["ERR."].values

# Funzione di fit: Coseno quadrato
def cos2(x, A, mu):
    return A * np.cos(np.radians(x - mu))**2

# Stima iniziale dei parametri (aiuta il fit a convergere!)
A0 = np.max(y)
mu0 = x[np.argmax(y)]  # Centro del picco
p0 = [A0, mu0]

# Maschera per rimuovere i dati NaN e infiniti
valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(y_err)) & (~np.isinf(y_err))

x_clean = x[valid_mask]
y_clean = y[valid_mask]
y_err_clean = y_err[valid_mask]

# Esegui il fit
popt, pcov = curve_fit(cos2, x_clean, y_clean, sigma=y_err_clean, p0=p0, absolute_sigma=True)

# Estrazione dei parametri e incertezze
A_fit, mu_fit = popt
dA, dmu = np.sqrt(np.diag(pcov))

print(f"Fit con funzione COS^2 trovato:")
print(f"A     = {A_fit:.3f} ± {dA:.3f}")
print(f"mu    = {mu_fit:.3f} ± {dmu:.3f}")

# Calcolare la bontà del fit (Chi-squared)
residuals = y_clean - cos2(x_clean, *popt)  # Differenza tra i dati e il fit
chi_squared = np.sum((residuals / y_err_clean) ** 2)
dof = len(x_clean) - len(popt)  # gradi di libertà
chi_squared_red = chi_squared / dof  # Chi-squared ridotto

print(f"Chi-squared: {chi_squared:.3f}")
print(f"Chi-squared ridotto: {chi_squared_red:.3f} (gradi di libertà: {dof})")

# Genera i valori di fit per il grafico
x_fit = np.linspace(-90, 90, 500)  # Solo tra -90 e +90
y_fit = cos2(x_fit, *popt)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.errorbar(x, y, yerr=y_err, fmt='o', label='Dati', capsize=5)
plt.plot(x_fit, y_fit, label='Fit COS^2', color='red')
plt.xlabel("Angolo [°]")
plt.ylabel("Flusso [Hz]")
plt.title("Fit COS^2 del Flusso")
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  # opzionale, se vuoi invertire l'asse X
plt.tight_layout()
plt.show()

# %%
