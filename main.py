import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

index_data = pd.read_csv('.\Indexes by Year.csv')

st.title("Trabalho de criação de um  aplicativo WEB usando Straemlit")
st.write(" Neste projeto estaremos trabalhando com um indice de classificação dos paises, indicando diversos indices desses países.\n")
st.write(" Vamos tentar criar um modelo que analise esses dados que são refletidos no indice de qualidade de vida baseando-se nos recursos obtidos dos países durante esses anos.\n")
st.write(" Este conjunto de dados contém diverrsos recursos, como por exemplo:\n")
st.write( " Rank: O rank desse país.\n")
st.write(" Country: O nome do país.\n")
st.write(" Quality of Life Index: O indice de Qualidade de vida.\n")
st.write(" Purchasing Power Index: O indice do poder de Compra.\n")
st.write(" Safety Index: O indice de segurança.\n")
st.write(" Year : O ano. ")
st.write(" Abaixo temos as Checkbox para poder ver a base de dados, os gráficos e por fim a previsão")
st.write(" Na esquerda pode ser alterado o test-size e o random-state para mexer no valor previsto")



check_index = st.checkbox('Apresentar Indices')

if check_index:
    st.write("Indices por anos")
    index_data


check_graf = st.checkbox('Apresentar Gráficos')


X = index_data.drop(["Climate Index", "Year", "Country", "Rank", ], axis=1)
y = index_data["Year"]

if check_graf:
    st.write("indices em gráfico")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.area_chart(X)
    df = pd.DataFrame(index_data, columns=['Rank', 'Health Care Index', 'Purchasing Power Index', 'Pollution Index'])
    df.hist()
    st.pyplot()


entrada = st.sidebar.slider("Test-Size", 0.1, 0.99, 0.33)
entrada2 = st.sidebar.slider("random_state", 1, 100, 63)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=entrada, random_state=entrada2)

logreg = LogisticRegression(solver='lbfgs', max_iter=10000)

logreg.fit(X_train, y_train)

prediction = logreg.predict(X_test)

check_pred = st.checkbox('Apresentar Predição')

if check_pred:
    st.write("Resultado da Regressão Logistica (altere o Test-Size e/ou o Random-State)")
    st.write(classification_report(y_test, prediction))


