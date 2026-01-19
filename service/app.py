import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
import os

# --- 1. Configuração de Caminhos ---
service_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(service_dir)
MODEL_PATH = os.path.join(project_root, "models_output", "modelo_adult_income.pkl")

# Variável global para o modelo
ml_models = {}

# --- 2. Lifespan (Inicialização) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carregar modelo
    try:
        if os.path.exists(MODEL_PATH):
            ml_models["model"] = joblib.load(MODEL_PATH)
            print(f"\n✅ SUCESSO: Modelo carregado de: {MODEL_PATH}")
        else:
            print(f"\n❌ ERRO: Arquivo não encontrado em: {MODEL_PATH}")
            ml_models["model"] = None
    except Exception as e:
        print(f"❌ Erro ao ler pickle: {e}")
        ml_models["model"] = None
    
    # --- MENSAGEM COM LINK CLICÁVEL ---
    print("\n" + "="*50)
    print("🚀 API ONLINE! ACESSE A DOCUMENTAÇÃO CLICANDO AQUI:")
    print("👉 http://localhost:8000/docs")
    print("="*50 + "\n")
    
    yield  # A API roda aqui
    
    ml_models.clear()

app = FastAPI(
    title="API Adult Income",
    description="Predição de renda (>50k ou <=50k) - Trabalho Final",
    version="1.0",
    lifespan=lifespan
)

# --- 3. Dados de Entrada ---
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
    return {"status": "Online", "docs": "http://localhost:8000/docs"}

@app.post("/predict")
def predict(data: IncomeInput):
    model = ml_models.get("model")
    if not model:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")
    
    try:
        input_data = data.model_dump(by_alias=True)
        df_input = pd.DataFrame([input_data])
        
        prediction = model.predict(df_input)
        
        # Converte numpy para nativo Python
        resultado_final = prediction[0]
        if hasattr(resultado_final, "item"):
            resultado_final = resultado_final.item()
        
        proba = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_input)
            proba = float(probs.max())

        return {
            "predicao": str(resultado_final),
            "confianca": f"{proba:.2%}"
        }
    except Exception as e:
        print(f"Erro detalhado: {e}")
        raise HTTPException(status_code=400, detail=f"Erro no processamento: {str(e)}")

if __name__ == "__main__":
    # Log level 'info' garante que nossos prints apareçam
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")