import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler,OneHotEncoder,LabelEncoder

def tratamento(adulto:pd.DataFrame):
    dataset = adulto.copy()

    dataset.drop(columns=['fnlwgt','education'],inplace=True)
    dataset.dropna(inplace=True)
    dataset.drop_duplicates(inplace=True)

    colunas_numericas = ['age','educational-num','capital-gain','capital-loss','hours-per-week']
    colunas_categoricas = ['workclass','marital-status','occupation','relationship','race','gender','native-country']

    Y = LabelEncoder().fit_transform(dataset['income'])
    X = dataset.drop(columns=['income'])

    pipe_num = Pipeline(steps=[('scaler',RobustScaler())])

    pipe_cat = Pipeline(steps=[('OHE',OneHotEncoder(handle_unknown='ignore'))])

    tratamento = ColumnTransformer(transformers=[('num', pipe_num, colunas_numericas),
                                                ('nom', pipe_cat, colunas_categoricas)])

    X_tratado = tratamento.fit_transform(X)

    return X_tratado,Y