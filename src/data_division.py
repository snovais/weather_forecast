import pandas as pd

import dask.dataframe as dd


def load_data():
    path = '/mnt/Sparetrack/weather/weather/data/brasilia2018_abril2021_in_use.csv'

    try:
        return pd.read_csv(path)
    except ValueError:
        raise ValueError("Falha em ler arquivo csv!")


def city_brasilia( db ):
    cd = db.to_numpy()
    list1 = []

    for i in range(len(db)):
        if cd[i, 19] == 'BRASILIA':
            list1.append(cd[i])

    return pd.DataFrame(list1, columns = [ 
                                'PRECIPITACAO_TOTAL_HORARIO_mm', 
                                'PRESSAO_ATMOSFERICA_NIVEL_DA_ESTACAO_HORARIA_mB',
                                'PRESSAO_ATMOSFERICA_MAX_NA_HORA_ANT_AUT_mB',
                                'PRESSAO_ATMOSFERICA_MIN_NA_HORA_ANT_AUT_mB', 
                                'RADIACAO_GLOBAL', 
                                'TEMPERATURA_DO_AR_BULBO_SECO_HORARIA',
                                'TEMPERATURA_PONTO_DE_ORVALHO', 
                                'TEMPERATURA_MAX_NA_HORA_ANT_AUT',
                                'TEMPERATURA_MIN_NA_HORA_ANT_AUT',
                                'TEMPERATURA_ORVALHO_MAX_NA_HORA_ANT_AUT',
                                'TEMPERATURA_ORVALHO_MIN_NA_HORA_ANT_AUT',
                                'UMIDADE_REL_MAX_NA_HORA_ANT_AUT_porcent', 
                                'UMIDADE_REL_MIN_NA_HORA_ANT_AUT_porcent', 
                                'UMIDADE_RELATIVA_DO_AR_HORARIA_porcent',
                                'VENTO_DIRECAO_HORARIA', 
                                'VENTO_RAJADA_MAX',
                                'VENTO_VELOCIDADE_HORARIA',   
                                'REGIAO', 'UF', 'ESTACAO', 'CODIGO_WMO',
                                'LATITUDE', 'LONGITUDE', 'ALTITUDE', 
                                'DATA_FUNDACAO', 'TIMESTAMP', 'DIA SENO', 
                                'DIA COS', 'ANO SENO', 'ANO COS'], index = None)


def data_without_locations( db ):
    return db.loc[:, ['PRECIPITACAO_TOTAL_HORARIO_mm',
                    'PRESSAO_ATMOSFERICA_NIVEL_DA_ESTACAO_HORARIA_mB',
                    'PRESSAO_ATMOSFERICA_MAX_NA_HORA_ANT_AUT_mB',
                    'PRESSAO_ATMOSFERICA_MIN_NA_HORA_ANT_AUT_mB',
                    'TEMPERATURA_DO_AR_BULBO_SECO_HORARIA',
                    'TEMPERATURA_PONTO_DE_ORVALHO',
                    'TEMPERATURA_MAX_NA_HORA_ANT_AUT',
                    'TEMPERATURA_MIN_NA_HORA_ANT_AUT',
                    'TEMPERATURA_ORVALHO_MAX_NA_HORA_ANT_AUT',
                    'TEMPERATURA_ORVALHO_MIN_NA_HORA_ANT_AUT',
                    'UMIDADE_REL_MAX_NA_HORA_ANT_AUT_porcent',
                    'UMIDADE_REL_MIN_NA_HORA_ANT_AUT_porcent',
                    'UMIDADE_RELATIVA_DO_AR_HORARIA_porcent',
                    'VENTO_DIRECAO_HORARIA', 'VENTO_RAJADA_MAX', 
                    'VENTO_VELOCIDADE_HORARIA', 'DIA SENO', 'DIA COS',
                    'ANO SENO', 'ANO COS']]


def get_names_cols():
    return ['PRECIPITACAO_TOTAL_HORARIO_mm',
            'PRESSAO_ATMOSFERICA_NIVEL_DA_ESTACAO_HORARIA_mB',
            'PRESSAO_ATMOSFERICA_MAX_NA_HORA_ANT_AUT_mB',
            'PRESSAO_ATMOSFERICA_MIN_NA_HORA_ANT_AUT_mB',
            'TEMPERATURA_DO_AR_BULBO_SECO_HORARIA',
            'TEMPERATURA_PONTO_DE_ORVALHO',
            'TEMPERATURA_MAX_NA_HORA_ANT_AUT',
            'TEMPERATURA_MIN_NA_HORA_ANT_AUT',
            'TEMPERATURA_ORVALHO_MAX_NA_HORA_ANT_AUT',
            'TEMPERATURA_ORVALHO_MIN_NA_HORA_ANT_AUT',
            'UMIDADE_REL_MAX_NA_HORA_ANT_AUT_porcent',
            'UMIDADE_REL_MIN_NA_HORA_ANT_AUT_porcent',
            'UMIDADE_RELATIVA_DO_AR_HORARIA_porcent',
            'VENTO_DIRECAO_HORARIA', 'VENTO_RAJADA_MAX', 
            'VENTO_VELOCIDADE_HORARIA', 'DIA SENO', 'DIA COS',
            'ANO SENO', 'ANO COS']


def db_division( db ):
    n = len(db)
    train_db = db[0:int(n*0.7)]
    val_db = db[int(n*0.07):int(n*0.9)]
    test_db = db[int(n*0.9):]

    num_features = db.shape[1]

    return train_db, val_db, test_db, num_features


def data_normalize( train_db, val_db, test_db ):
    train_mean = train_db.mean()
    train_std = train_db.std()
    
    train_db = (train_db - train_mean)/train_std
    val_db = (val_db - train_mean)/train_std
    test_db = (test_db - train_mean)/train_std 
    """
    train_db = (train_db - train_db.min())/(train_db.max() - train_db.min())
    val_db = (val_db - train_db.min())/(train_db.max() - train_db.min())
    test_db = (test_db - train_db.min())/(train_db.max() - train_db.min())
    """
    return train_db, val_db, test_db, train_mean, train_std
