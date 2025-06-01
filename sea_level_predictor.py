import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
import numpy as np

def draw_plot():
    # Read data from file
    df = pd.read_csv('epa-sea-level.csv')


    # Create scatter plot
    plt.xlabel('Year')
    plt.ylabel('CSIRO Adjusted Sea Level')
    plt.scatter(x=df['Year'], y=df['CSIRO Adjusted Sea Level'])

    # Create first line of best fit
    df1 = df.drop('NOAA Adjusted Sea Level', axis=1)

    """ Limpiando la columna de NOAA llena de valores nulos """
    df = df.drop('NOAA Adjusted Sea Level', axis=1)
    def train_val_test_split (df, rstate=42, shuffle=True, stratify=None):
        """ Funcion para dividir los datos """
        
        if stratify != None:
            strat = df[stratify]
        else:
            strat=None
        
        train_set, test_set = tts(df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
        if stratify != None:
            strat = test_set[stratify]
        else:
            strat=None
        test_set, val_set = tts(test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
        return (train_set, val_set, test_set)

    train_set, val_set, test_set = train_val_test_split(df, rstate=42)
    # Para el conjunto de datos general
    x_df = df.drop(['CSIRO Adjusted Sea Level', 'Lower Error Bound', 'Upper Error Bound'], axis=1)
    y_df = df['CSIRO Adjusted Sea Level'].copy()

    # Para el conjunto de datos de entrenamiento
    x_train = train_set.drop(['CSIRO Adjusted Sea Level', 'Lower Error Bound', 'Upper Error Bound'], axis=1)
    y_train = train_set['CSIRO Adjusted Sea Level'].copy()

    # Para el conjunto de datos de validacion 
    x_val = val_set.drop(['CSIRO Adjusted Sea Level', 'Lower Error Bound', 'Upper Error Bound'], axis=1)
    y_val = val_set['CSIRO Adjusted Sea Level'].copy()

    # Conjunto de datos de pruebas
    x_test = test_set.drop(['CSIRO Adjusted Sea Level', 'Lower Error Bound', 'Upper Error Bound'], axis=1)
    y_test = test_set['CSIRO Adjusted Sea Level'].copy()

    """ Creacion del algoritmo """
    linreg = LinearRegression()
    reg = linreg.fit(x_train, y_train)
    
    """ Prediccion de nuevos ejemplos """
    y_pred = reg.predict(x_val)

    """ Analizando la precision """
    prec = reg.score(x_train, y_train)

    """ Prediciendo para el 2050 """
    prediction = linreg.predict(np.array([[2050]]))

    """ Dibujando la prediccion """
    plt.scatter(x=df1['Year'], y=df1['CSIRO Adjusted Sea Level'], color='red', s=3)
    plt.plot([df1['Year'][0], 2050], [df1['CSIRO Adjusted Sea Level'][0], prediction[0]], c='blue')



    # Create second line of best fit
    """ Making the second line df """
    dfsl = df.copy()
    dfsl['CSIRO Adjusted Sea Level'] = dfsl['CSIRO Adjusted Sea Level'][120:]
    dfsl['Year'] = df['Year'][120:]
    dfsl = dfsl.dropna()
    dfsl = dfsl.drop(['Lower Error Bound', 'Upper Error Bound', 'NOAA Adjusted Sea Level'], axis=1)


    train_set, val_set, test_set = train_val_test_split(dfsl, rstate=42)
    # Para el conjunto de datos general
    x_df = dfsl.drop(['CSIRO Adjusted Sea Level'], axis=1)
    y_df = dfsl['CSIRO Adjusted Sea Level'].copy()

    # Para el conjunto de datos de entrenamiento
    x_train = train_set.drop(['CSIRO Adjusted Sea Level'], axis=1)
    y_train = train_set['CSIRO Adjusted Sea Level'].copy()

    # Para el conjunto de datos de validacion 
    x_val = val_set.drop(['CSIRO Adjusted Sea Level'], axis=1)
    y_val = val_set['CSIRO Adjusted Sea Level'].copy()

    # Conjunto de datos de pruebas
    x_test = test_set.drop(['CSIRO Adjusted Sea Level'], axis=1)
    y_test = test_set['CSIRO Adjusted Sea Level'].copy()

    """ Creacion del algoritmo """
    linreg = LinearRegression()
    reg = linreg.fit(x_train, y_train)

    """ Prediccion de nuevos ejemplos """
    y_pred = reg.predict(x_val)

    """ Analizando la precision """
    prec = reg.score(x_train, y_train)
    print(prec)
    """ Prediciendo para el 2050 """
    prediction = linreg.predict(np.array([[2050]]))

    """ Dibujando la prediccion """
    plt.title('')
    plt.scatter(x=dfsl['Year'], y=dfsl['CSIRO Adjusted Sea Level'], color='red', s=3)
    plt.plot([dfsl['Year'][120], 2050], [dfsl['CSIRO Adjusted Sea Level'][120], prediction[0]], c='blue')
    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')

    # Add labels and title

    
    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plot.png')
    return plt.gca()