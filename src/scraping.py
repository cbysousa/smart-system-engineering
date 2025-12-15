import kaggle
import pandas as pd
import json

# download do dataset do kaggle
kaggle.api.dataset_download_files(
    "mragpavank/breast-cancer",
    path=".",
    unzip=True
)

df = pd.read_csv("data.csv")

# converte para lista de registros (scraping dos dados)
records = df.to_dict(orient="records")

# transforma o csv em um json
with open("breast_cancer.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)

print("✔ Scraping concluído e JSON gerado!")

# -----------------------------------------
# remoção da coluna vazia no final do json
# -----------------------------------------

# carrega o JSON gerado pelo scraping
with open("breast_cancer.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# remove a chave "Unnamed: 32" de cada registro
for item in data:
    item.pop("Unnamed: 32", None)  # None evita erro se não existir

# salva novamente
with open("breast_cancer.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
