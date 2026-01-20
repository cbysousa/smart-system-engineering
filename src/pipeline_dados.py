import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    
    dataset.drop(columns=['fnlwgt','education'],inplace=True)
    dataset.dropna(inplace=True)
    dataset.drop_duplicates(inplace=True)

    dataset['income'] = LabelEncoder().fit_transform(dataset['income'])
    
    pasta_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_saida = os.path.join(pasta_atual, 'adult_limpo.csv')
    
    dataset.to_csv(caminho_saida, index=False)

def main():
    adult = pd.read_csv("dados/adult.csv", na_values='?')
    tratamento(adult)

if __name__ == '__main__':
    main()