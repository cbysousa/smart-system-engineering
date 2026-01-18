import kaggle
import pandas as pd
import json
import os

# identifica a pasta onde este script (scraping.py) está localizado (pasta src)
src_folder = os.path.dirname(os.path.abspath(__file__))

# define os caminhos completos para os arquivos
csv_path = os.path.join(src_folder, "adult.csv")
json_path = os.path.join(src_folder, "adult.json")

# download do dataset do kaggle
kaggle.api.dataset_download_files(
    "wenruliu/adult-income-dataset",
    path=src_folder,
    unzip=True
)

df = pd.read_csv("adult.csv", na_values='?')

# converte para lista de registros (scraping dos dados)
records = df.to_dict(orient="records")

# transforma o csv em um json
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=4, ensure_ascii=False)

print("✔ Scraping concluído e JSON gerado!")

# carrega o JSON gerado pelo scraping
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)