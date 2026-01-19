import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

# importando csv e transformando em dataset
path_dataset = "dados/adult_processado.csv"
df = pd.read_csv(path_dataset)

# definindo coluna alvo, e dividindo em treino e teste
coluna_alvo = "income"
X = df.drop(columns=[coluna_alvo])
Y = df[coluna_alvo]
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)


def model_random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10) # limitando a altura da árvore, afim de evitar overfitting

def model_knn():
    return KNeighborsClassifier(n_neighbors=5)

def model_regressao_logistica():
    return LogisticRegression(max_iter=2000, random_state=42, solver='liblinear')

def model_xgboost():
    return GradientBoostingClassifier(random_state=42)

modelos = {
    "Logistic Regression": model_regressao_logistica(),
    "KNN": model_knn(),
    "Random Forest": model_random_forest(),
    "XGBoost (Gradient)": model_xgboost()
}

# Define 5 divisões (folds), embaralhando os dados antes
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lista de métricas que queremos calcular de uma vez
metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

print("\n--- INICIANDO VALIDAÇÃO CRUZADA (STRATIFIED K-FOLD) ---\n")

for nome, modelo in modelos.items():
    print(f"Avaliando {nome}")
    
    # O cross_validate faz tudo: divide, treina e avalia 5 vezes
    # Atenção: Passamos X e Y inteiros aqui, não X_treino/X_teste
    resultados = cross_validate(modelo, X, Y, cv=skf, scoring=metricas)
    
    # Os resultados vêm como listas (ex: 5 acurácias). Vamos pegar a média e o desvio padrão.
    acuracia_media = resultados['test_accuracy'].mean()
    f1_media = resultados['test_f1'].mean()
    precision = resultados['test_precision'].mean()
    recall = resultados['test_recall'].mean()
    auc_media = resultados['test_roc_auc'].mean()
    
    print(f"Resultados Médios para {nome}:")
    print(f" >> Acurácia: {acuracia_media:.4f} (± {resultados['test_accuracy'].std():.4f})")
    print(f" >> Precision: {precision:.4f} (± {resultados['test_precision'].std():.4f})")
    print(f" >> Recall:    {recall:.4f} (± {resultados['test_recall'].std():.4f})")
    print(f" >> F1-Score: {f1_media:.4f} (± {resultados['test_f1'].std():.4f})")
    print(f" >> ROC-AUC:  {auc_media:.4f}")
    print("-" * 40)
    
print("\n--- GERANDO GRÁFICO DA CURVA ROC ---\n")

for nome, modelo in modelos.items():
    
    # Precisamos treinar o modelo nos dados de treino para testar no X_teste
    # (O K-Fold anterior treinou e descartou, então treinamos de novo aqui para o gráfico)
    modelo.fit(X_treino, Y_treino)
    
    if hasattr(modelo, "predict_proba"):
        # Pega a probabilidade da classe 1 (>50k)
        y_prob = modelo.predict_proba(X_teste)[:, 1]
        
        # Calcula os pontos da curva
        fpr, tpr, _ = roc_curve(Y_teste, y_prob)
        auc_score = auc(fpr, tpr)
        
        # Plota a linha
        plt.plot(fpr, tpr, linewidth=2, label=f'{nome} (AUC = {auc_score:.3f})')
        print(f"Plotando curva para: {nome}")

# 4. Finalização e Salvamento
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (False Positive Rate)', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos (Recall)', fontsize=12)
plt.title('Comparação da Curva ROC - Modelos de Renda', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

# Salva o arquivo
plt.savefig('grafico_curvaROC.jpg')

print("\n--- VERIFICAÇÃO DE OVERFITTING ---\n")

for nome, modelo in modelos.items():
    # 1. Treina o modelo
    modelo.fit(X_treino, Y_treino)
    
    # 2. Previsão na base de TREINO (Memorização)
    # Se tiver predict_proba (ideal para AUC), usa ele. Se não, usa predict.
    if hasattr(modelo, "predict_proba"):
        y_prob_treino = modelo.predict_proba(X_treino)[:, 1]
        y_prob_teste = modelo.predict_proba(X_teste)[:, 1]
        metric_treino = roc_auc_score(Y_treino, y_prob_treino)
        metric_teste = roc_auc_score(Y_teste, y_prob_teste)
        nome_metrica = "ROC-AUC"
    else:
        # Fallback para acurácia se não tiver probabilidade
        y_pred_treino = modelo.predict(X_treino)
        y_pred_teste = modelo.predict(X_teste)
        metric_treino = accuracy_score(Y_treino, y_pred_treino)
        metric_teste = accuracy_score(Y_teste, y_pred_teste)
        nome_metrica = "Acurácia"

    # 3. Calcula a diferença (Gap)
    diferenca = metric_treino - metric_teste
    
    print(f"Analisando {nome}:")
    print(f" >> {nome_metrica} no TREINO: {metric_treino:.4f}")
    print(f" >> {nome_metrica} no TESTE:  {metric_teste:.4f}")
    print(f" >> Diferença (Gap):  {diferenca:.4f}")
    
    if diferenca > 0.10: # Se a diferença for maior que 10%
        print(" ⚠️  ALERTA DE OVERFITTING GRAVE!")
    elif diferenca > 0.05: # Se for maior que 5%
        print(" ⚠️  Sinal de alerta (leve overfitting)")
    else:
        print(" ✅ Modelo estável (bom generalista)")
        
    print("-" * 30)
