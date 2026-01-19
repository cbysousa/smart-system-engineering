`# Smart System Engineering
This repository hosts the final project developed for the Smart Systems Engineering course, taught by Lincoln Rocha at the Federal University of Ceará (UFC)`

# Trabalho Final - Engenharia de Sistemas Inteligentes

Projeto de classificação (Machine Learning) utilizando o dataset **Adult Income**, com foco em engenharia de software e reprodutibilidade. O sistema prevê se a renda de uma pessoa excede 50k/ano com base em dados demográficos.

## 📋 Estrutura do Projeto
O trabalho foi dividido em três módulos principais:
- **Parte 1: Pipeline de Dados** (Extração, limpeza e análise exploratória).
- **Parte 2: Pipeline de Modelos** (Treinamento, validação cruzada e geração do modelo serializado `.pkl`).
- **Parte 3: Módulo de Serviço** (API REST com FastAPI encapsulada via Docker).

---

## 🚀 Como Rodar (Modo Docker) - Recomendado
Para rodar a aplicação em qualquer sistema operacional (Linux, Windows, Mac) sem precisar instalar Python ou dependências, utilizamos o **Docker**.

**1. Construir a imagem (Build):**
Na raiz do projeto, execute:
```bash
docker build -t api-trabalho-final -f service/Dockerfile .
```
**2. Rodar o container (Run):**
```bash
docker run -p 8000:8000 api-trabalho-final
```
**3. Testar a API:** 
Acesse a documentação interativa no navegador: 👉 [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs&authuser=1)
## 🔧 Como Rodar (Modo Desenvolvimento Local)

Caso queira rodar diretamente na máquina utilizando **Poetry**:

**1. Instale as dependências:**
```Bash
poetry install
```
**2. Execute o servidor:**
```bash
poetry run python service/app.py
```

---

## 🧪 Exemplo de Requisição

Para testar a rota `POST /predict`, você pode usar o seguinte JSON de exemplo:
```JSON
{
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
```

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.9+
	
- **Gerenciamento de Pacotes:** Poetry
    
- **ML & Dados:** Scikit-Learn, Pandas, Numpy, Joblib
    
- **API:** FastAPI, Uvicorn, Pydantic
    
- **DevOps:** Docker