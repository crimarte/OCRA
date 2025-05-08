import pandas as pd
import re

# --- Lettura CSV originale ---
raw = pd.read_csv("Misura flusso vs angolo_CRC_OCRA_PCTO2025(Muon FLUX DATA).csv",
                  sep=",", decimal=".", encoding="cp1252", header=None)

# Estrai header e dati
header_raw = raw.iloc[0].tolist()
data = raw.iloc[1:].reset_index(drop=True)

# Pulizia nomi colonne
def normalizza_nome(col):
    if not isinstance(col, str):
        return col
    col = col.encode("ascii", "ignore").decode("utf-8")
    col = re.sub(r"[°øØ´`'’]", "", col)
    return col.strip()

header_puliti = [normalizza_nome(col) for col in header_raw]
data.columns = header_puliti

# Verifica numero colonne
if len(data.columns) < 19:
    raise ValueError("❌ Il file non ha almeno 19 colonne (ci vogliono 0, 17, 18)")

# --- Seleziona solo le colonne 0, 17 e 18 ---
df = data.iloc[:, [0, 17, 18]].copy()
df.columns = ["Angel []", "Flux [Hz]", "ERR."]

# --- Conversione numerica ---
df["Angel []"] = pd.to_numeric(df["Angel []"], errors="coerce")
df["Flux [Hz]"] = pd.to_numeric(df["Flux [Hz]"], errors="coerce")
df["ERR."] = pd.to_numeric(df["ERR."], errors="coerce")

# --- Salvataggio CSV pulito ---
df.dropna().to_csv("Misura_flusso_ripulito.csv", index=False, encoding="utf-8")

# --- Output verifica ---
print(df.head(10))
print("✅ CSV pulito salvato come 'Misura_flusso_ripulito.csv'")
