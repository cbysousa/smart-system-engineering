# Usa uma imagem leve do Python
FROM python:3.9-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# 1. Copia e instala as dependências
# (Copiamos primeiro para aproveitar o cache do Docker)
COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copia a pasta do código da API
COPY service/ service/

# 3. Copia a pasta do modelo
# Isso garante que a estrutura interna fique igual à do seu computador
COPY models_output/ models_output/

# Expõe a porta 8000
EXPOSE 8000

# Comando para rodar a API quando o container iniciar
# Note que ajustamos o caminho do app para 'service.app:app' ou rodamos de dentro
CMD ["python", "service/app.py"]