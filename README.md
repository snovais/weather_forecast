Pesquisa e testes com experimentos utilizando redes neurais simples, densas, convolucionais, recorrentes e convolucionais recorrentes para previsão de um único passo, isto é, quando se deseja prever apenas informações num determinado ponto do futuro. Além disso, consta algoritmos para previsão de multi pontos do tempo, isto é, realizar previsão de temperatura, velocidade do vento e precipitação para as próximas 120 horas a fim de avaliar a possibilidade de grandes alterações. Para isto, foram selecionados dados dos últimos três de uma uníca cidade (Brasília) para buscar criar um algoritmo robusto, na qual seria possível após acerto do modelo replicar para N cidades.

Detalhamento:
Os algoritmos sempre rodam com os dados pré-processados e depois fazem uso do divisor de dados (data_division.py), um segundo algoritmo para criar as o modelos de dados em janelas (windows generator.py) e um algoritmo profundo (rede reural).
Exemplo:
    conv_model tem como dependência data_division e window_generator.


Instalar vscode + python +3.7 + Anaconda https://www.anaconda.com/products/individual

Bibliotecas necessárias:

numpy = pip install numpy

pandas = pip install pandas

tensorflow = pip install tensorflow

joblib = pip install joblib

matplotlib = pip install matplotlib

seaborn = pip install seaborn

# RUN: gru_model_multi_output_v3.py# weather_forecast
