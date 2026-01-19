import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate

# importando csv e transformando em dataset
path_dataset = "dados/adult_limpo.csv"
df = pd.read_csv(path_dataset)

# definindo coluna alvo, e dividindo em treino (70%) e teste (30%) ; estratificação mantém proporção das classes
coluna_alvo = "income"
X = df.drop(columns=[coluna_alvo])
Y = df[coluna_alvo]

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# separação de colunas numéricas e categóricas 
colunas_numericas = ['age','educational-num','capital-gain','capital-loss','hours-per-week']
colunas_categoricas = ['workclass','marital-status','occupation','relationship','race','gender','native-country']

# pré-processamento
processor = ColumnTransformer(transformers=[
    ('num', RobustScaler(), colunas_numericas), # escala os dados numéricos de acordo com mediana e intervalo interquartil
    ('cat', OneHotEncoder(handle_unknown='ignore'), colunas_categoricas)  # transformando dados categóricos em variáveis binárias
])

def get_pipelines():
    
    pipelines = {}
    
    # Random forest 
    ''' 
        Ajuste de hiperparâmetros:
    - n_estimators=100 -> número balanceado de árvores geradas
    - max_depth=12 -> profundidade máxima de árvores 
    - class_weight = 'balanced' -> dá mais peso à classe minoria (>50k)
    
    '''
    pipelines["Random Forest"] = Pipeline([
        ('preprocessor', processor), 
        ('model', RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42))
    ])
    
    # KNN 
    '''  Ajuste de hiperparâmetros:
    - n_neighbors=50 -> quantidade de vizinhos ; valores mais baixos causaram overfitting. ao aumentar o valor, tivemos resultados mais positivos
    - weights='uniform' -> todos os vizinhos possuem o mesmo peso.'''
    pipelines["KNN"] = Pipeline([
        ('preprocessor', processor),
        ('model', KNeighborsClassifier(n_neighbors=50, weights='uniform')) 
    ])
    
    # Regressão Logística 
    ''' 
        Ajuste de hiperparâmetros:
    - max_iter = 2000 -> O dataset é relativamente grande (>40k linhas), além de possuir muitas features, o que pede um maior número de iterações 
    - solver = 'liblinear' -> 
    - class_weight='balanced' -> ajusta os pesos de forma inversamente proporcional à frequência das classes ; como vimos na AED, a classe >50k é MINORIA!
    '''
    pipelines["Regressão Logística"] = Pipeline([
        ('preprocessor', processor),
        ('model', LogisticRegression(max_iter=2000, random_state=42, solver='liblinear', class_weight='balanced'))
    ])
    
    # Gradient Boost
    pipelines["Gradient Boost"] = Pipeline([
        ('preprocessor', processor),
        ('model', GradientBoostingClassifier(random_state=42))
    ])
    
    return pipelines

modelos = get_pipelines()

# divisão do conjunto de treino em 5 partes, evitando possíveis ordenações  
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

# métricas utilizadas para análise dos modelos
metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

print("\n====== Validação cruzada ======\n")

for nome, modelo in modelos.items():
    print(f"\nAvaliando {nome}")
    
    resultados = cross_validate(modelo, X_treino, Y_treino, cv=skf, scoring=metricas)
    
    # desempenho médio dos modelos, com as métricas escolhidas
    print(f"Desempenho médio - {nome}:")
    print(f" >> ROC-AUC:  {resultados['test_roc_auc'].mean():.4f}")
    print(f" >> Acurácia: {resultados['test_accuracy'].mean():.4f}")
    print(f" >> Precision: {resultados['test_precision'].mean():.4f}")
    print(f" >> Recall:    {resultados['test_recall'].mean():.4f}")
    print(f" >> F1-Score: {resultados['test_f1'].mean():.4f}")
    print("-" * 30)


print("\n====== Treinamento final + Geração de gráfico de curva ROC ======\n")


plt.figure(figsize=(10, 8)) # tamanho do gráfico
for nome, modelo in modelos.items():
    
    # treinamento do modelo
    modelo.fit(X_treino, Y_treino)
    
    if hasattr(modelo, "predict_proba"):
        y_prob_teste = modelo.predict_proba(X_teste)[:, 1]
        y_prob_treino = modelo.predict_proba(X_treino)[:, 1] # checagem de overfitting
        
        # plotando curva ROC
        fpr, tpr, _ = roc_curve(Y_teste, y_prob_teste)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{nome} (AUC = {auc_score:.3f})')
        
        # cálculo de possível overfitting dos modelos
        metric_treino = roc_auc_score(Y_treino, y_prob_treino)
        metric_teste = roc_auc_score(Y_teste, y_prob_teste)
        nome_metrica = "ROC-AUC"
        
    else:
        # fallback (caso algum modelo futuro não tenha predict_proba)
        y_pred_teste = modelo.predict(X_teste)
        y_pred_treino = modelo.predict(X_treino)
        metric_treino = accuracy_score(Y_treino, y_pred_treino)
        metric_teste = accuracy_score(Y_teste, y_pred_teste)
        nome_metrica = "Acurácia"

    # análise de overfitting dos modelos
    gap = metric_treino - metric_teste
    print(f"[{nome}] Análise de Generalização - ({nome_metrica}):")
    print(f"   Treino: {metric_treino:.4f} | Teste: {metric_teste:.4f} | Gap: {gap:.4f}")
    
    if gap > 0.10:
        print("ALTO OVERFITTING")
    elif gap > 0.05:
        print("LEVE OVERFITTING")
    else:
        print("Modelo estável!")
    print("-" * 30)
    print('\n')

# configuração final do gráfico
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos (Recall)', fontsize=12)
plt.title('Curva ROC - Comparação de Modelos', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.savefig('grafico_curvaROC.jpg')

# ==========================================
# BLOCO FINAL: RELATÓRIO E SALVAMENTO (ATUALIZADO)
# ==========================================
import joblib
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

print("\n====== Gerando Relatório e Salvando Modelo (Parte 3) ======\n")

# Configurando pasta de saída
pasta_destino = 'models_output'
if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

# Caminhos dos arquivos
arquivo_relatorio = os.path.join(pasta_destino, 'relatorio_performance.txt')
arquivo_modelo = os.path.join(pasta_destino, 'modelo_adult_income.pkl')

# 1. Gerando o Relatório de Performance
print(f"Comparando modelos e salvando em: {arquivo_relatorio} ...")

with open(arquivo_relatorio, "w") as f:
    f.write("=========================================\n")
    f.write("   RELATÓRIO DE PERFORMANCE DOS MODELOS  \n")
    f.write("=========================================\n\n")
    
    # Itera sobre todos os modelos que você testou
    for nome, modelo in modelos.items():
        # Vamos fazer uma validação rápida (cross-validation) para ter uma métrica robusta
        # cv=5 significa que ele testa 5 vezes e tira a média
        scores = cross_val_score(modelo, X, Y, cv=5, scoring='accuracy')
        media_acc = scores.mean() * 100
        
        linha = f"Modelo: {nome:<20} | Acurácia Média: {media_acc:.2f}%\n"
        print(linha.strip()) # Mostra no terminal
        f.write(linha)       # Salva no arquivo

    f.write("\n=========================================\n")
    f.write("Modelo Vencedor escolhido para Produção: Gradient Boost\n")
    f.write("=========================================\n")

# 2. Treinando e Salvando o Campeão (Gradient Boost)
print(f"\nSalvando o modelo campeão em: {arquivo_modelo} ...")

modelo_final = modelos["Gradient Boost"]
modelo_final.fit(X, Y) # Treina com tudo antes de salvar
joblib.dump(modelo_final, arquivo_modelo)

print(f"✅ Sucesso! Relatório e Modelo salvos na pasta '{pasta_destino}'.")