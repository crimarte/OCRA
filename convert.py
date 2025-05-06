import pandas as pd
import unicodedata

# --- Step 1: Lettura CSV originale ---
raw = pd.read_csv("Misura flusso vs angolo_CRC_OCRA_PCTO2025_csv.csv", sep=";", decimal=",", encoding="cp1252", header=None)

# Estrai la prima riga come header
header = raw.iloc[0].tolist()
df = raw.iloc[1:].copy()
df = df.reset_index(drop=True)

# --- Step 2: Pulizia nomi colonne ---
def normalizza_nome(col):
    if not isinstance(col, str):
        return col
    # Rimuove caratteri speciali tipo ø o °
    col = unicodedata.normalize("NFKD", col).encode("ASCII", "ignore").decode("utf-8")
    # Rimuove spazi all'inizio/fine
    col = col.strip()
    return col

header_puliti = [normalizza_nome(col) for col in header]

# Applica i nuovi nomi al DataFrame
df.columns = header_puliti

# --- Step 3: Rimuove colonne non desiderate ---
colonne_da_rimuovere = [f"Misure {i}" for i in range(4, 11)]
df = df.drop(columns=[col for col in colonne_da_rimuovere if col in df.columns])

# --- Step 4: Salvataggio del file pulito ---
df.to_csv("Misura_flusso_ripulito.csv", index=False, encoding="utf-8")

print("✅ CSV pulito salvato come 'Misura_flusso_ripulito.csv'")
