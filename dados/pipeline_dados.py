import pandas as pd
# import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler,OneHotEncoder,LabelEncoder

def tratamento(adulto:pd.DataFrame):
    dataset = adulto.copy()
    '''
    A coluna 'education' é equivalente a coluna 'educational-num', só que com os graus
    de escolaridade como strings, ao invés de números, e como precisamos do valor numérico,
    optamos por removê-la e usar apenas a 'educational-num'.

    A coluna 'fnlwgt' representa a quantidade de pessoas naquele país que tem os dados exatamente
    iguais aos daquela linha, o que desbalanceia o modelo, por isso decidimos remover.

    Removemos as linhas com valores nulos, pois a porcentagem dessas linhas é baixa (cerca de 5%).

    Removemos também linhas duplicadas.
    '''
    # dataset.replace('?', np.nan(), inplace=True)
    dataset.drop(columns=['fnlwgt','education'],inplace=True)
    dataset.dropna(inplace=True)
    dataset.drop_duplicates(inplace=True)

    #Separação de colunas numéricas e categóricas.
    colunas_numericas = ['age','educational-num','capital-gain','capital-loss','hours-per-week']
    colunas_categoricas = ['workclass','marital-status','occupation','relationship','race','gender','native-country']

    #Separação da coluna alvo e das features, para fazer a transformação apenas nas features.
    Y = LabelEncoder().fit_transform(dataset['income'])
    X = dataset.drop(columns=['income'])

    #Pipelines de tratamento.
    pipe_num = Pipeline(steps=[('scaler',RobustScaler())])

    pipe_cat = Pipeline(steps=[('OHE',OneHotEncoder(handle_unknown='ignore'))])

    tratamento = ColumnTransformer(transformers=[('num', pipe_num, colunas_numericas),
                                                ('nom', pipe_cat, colunas_categoricas)])

    X_tratado = tratamento.fit_transform(X)
    
    nomes_colunas = tratamento.get_feature_names_out()
    
    try:
        df_final = pd.DataFrame(X_tratado.toarray(), columns=nomes_colunas)
    except AttributeError:
        df_final = pd.DataFrame(X_tratado, columns=nomes_colunas)
        
    df_final['income'] = Y
    
    pasta_atual = os.path.dirname(os.path.abspath(__file__))

    caminho_saida = os.path.join(pasta_atual, 'adult_processado.csv')

    df_final.to_csv(caminho_saida, index=False)

def main():
    adult = pd.read_csv("dados/adult.csv", na_values='?')
    tratamento(adult)
    path = "dados/adult_processado.csv"
    df = pd.read_csv(path)
    a = df.head()
    print(a)

if __name__ == '__main__':
    main()