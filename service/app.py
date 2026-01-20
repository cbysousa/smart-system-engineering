import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
import os

# Configuração dos diretórios e caminho do arquivo do modelo
service_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(service_dir)
MODEL_PATH = os.path.join(project_root, "models_output", "modelo_adult_income.pkl")

# Armazena o modelo carregado globalmente
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    Carrega o modelo na inicialização e libera recursos no encerramento.
    """
    # Tenta carregar o modelo .pkl
    try:
        if os.path.exists(MODEL_PATH):
            ml_models["model"] = joblib.load(MODEL_PATH)
            print(f"[INFO] Modelo carregado com sucesso: {MODEL_PATH}")
        else:
            print(f"[AVISO] Arquivo do modelo não encontrado em: {MODEL_PATH}")
            ml_models["model"] = None
    except Exception as e:
        print(f"[ERRO] Falha ao ler o arquivo pickle: {e}")
        ml_models["model"] = None
    
    # Exibe o link da documentação no terminal para facilitar o acesso
    print("-" * 60)
    print("API iniciada.")
    print("Acesse a documentação em: http://localhost:8000/docs")
    print("-" * 60)
    
    yield
    
    # Limpeza de recursos ao encerrar
    ml_models.clear()

app = FastAPI(
    title="API Adult Income",
    description="Serviço de predição de renda com base em dados demográficos.",
    version="1.0",
    lifespan=lifespan
)

# Definição do esquema de dados para validação (Pydantic)
class IncomeInput(BaseModel):
    age: int
    educational_num: int = Field(..., alias="educational-num")
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    workclass: str
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    gender: str
    native_country: str = Field(..., alias="native-country")

    # Configuração para aceitar alias e fornecer exemplo na doc
    model_config = ConfigDict(
        populate_by_name=True, 
        json_schema_extra={
            "example": {
                "age": 45,
                "educational-num": 13,
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 50,
                "workclass": "Private",
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "gender": "Male",
                "native-country": "United-States"
            }
        }
    )

@app.get("/")
def home():
    """Rota de verificação de status."""
    return {"status": "Online", "docs": "http://localhost:8000/docs"}

@app.post("/predict")
def predict(data: IncomeInput):
    """
    Recebe os dados demográficos e retorna a predição do modelo.
    """
    model = ml_models.get("model")
    if not model:
        raise HTTPException(status_code=500, detail="Modelo não disponível no servidor.")
    
    try:
        # Prepara os dados para o formato esperado pelo DataFrame
        input_data = data.model_dump(by_alias=True)
        df_input = pd.DataFrame([input_data])
        
        prediction = model.predict(df_input)
        
        # O resultado do numpy precisa ser convertido para tipo nativo do Python
        # para garantir a serialização correta no retorno JSON
        resultado_final = prediction[0]
        if hasattr(resultado_final, "item"):
            resultado_final = resultado_final.item()
        
        # Calcula a probabilidade/confiança se o modelo suportar
        proba = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_input)
            proba = float(probs.max())

        return {
            "predicao": str(resultado_final),
            "confianca": f"{proba:.2%}"
        }
    except Exception as e:
        print(f"Erro no processamento da requisição: {e}")
        raise HTTPException(status_code=400, detail=f"Erro no processamento: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")