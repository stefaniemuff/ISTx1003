# importere pakker og funksjoner vi trenger i oppgave 1

# generelt - numerikk og nyttige funksjoner
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt

import seaborn as sns

# Fordelinger, modeller for regresjon, qq-plott 
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as  sm

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
InteractiveShell.ast_node_interactivity = "last_expr"


# Oppgave 1a

# Lese inn datasettet ved funksjon fra pandas  (df=data frame - vanlig navn å gi et datasett)
df = pd.read_csv('~/Teaching/ISTT1003/2021/Slides/1Regresjon/data/bodyfat.clean.txt',sep="\t",index_col=None)

df = df[['bodyfat','age','weight','height','bmi','neck','abdomen','hip']]
 

df.sort_values(by=['bmi'],inplace=True) # alt sortert på Hoeyde, bare for gøy

# Skriv ut de første og siste radene
print(df)

# Some summary statistics
df.describe()

# Konverter kjønn og idrettsgren til kategory
#df=df.astype({'Kjoenn':'category','Sport':'category'})
#print(df["Kjoenn"].value_counts())
#print(df["Sport"].value_counts())


# Oppgave 1b, enkel linear regresjon

sns.relplot(x = 'bmi', y = 'bodyfat', kind = 'scatter', data = df)
plt.show()

# kodechunk Steg2-4

# Steg 2: spesifiser matematisk modell
# Formula for lin reg:
formel='bodyfat ~ bmi '

# Steg 3: Initaliser og tilpass en enkel lineær regresjonsmodell
# først initialisere
modell = smf.ols(formel,data=df)
# deretter tilpasse
resultat = modell.fit()

# Steg 4: Presenter resultater fra den tilpassede regresjonsmodellen
print(resultat.summary())


# Derive intercept b and slope m
# m, b = np.polyfit(df['bmi'],df['bodyfat'], 1)
# sns.relplot(x = 'bmi', y = 'bodyfat', kind = 'scatter', data = df)
# plt.plot(x, m*x + b)
# plt.show()

sns.regplot(df['bmi'],df['bodyfat'])


## Model checking

# kodechunk Steg5

# Steg 5: Evaluere om modellen passer til dataene
# Plotte predikert verdi mot residual 
sns.scatterplot(x=resultat.fittedvalues, y=resultat.resid)
plt.ylabel("Residual")
plt.xlabel("Predikert verdi")
plt.show()

# Lage kvantil-kvantil-plott for residualene
sm.qqplot(resultat.resid,line='45',fit=True)
plt.ylabel("Kvantiler i residualene")
plt.xlabel("Kvantiler i normalfordelingen")
plt.show()


## Multiple linear regression
# Kryssplott av Hoeyde mot Blodceller, Vekt mot Blodceller og Hoeyde mot Vekt.
# På diagonalen er glattede histogrammer (tetthetsplott) av  Blodceller, Hoeyde og Vekt



## Tilpass multippel linear regresjon (2 variabler):
    
formel='bodyfat ~ bmi + age'

# Steg 3: Initaliser og tilpass en enkel lineær regresjonsmodell
# først initialisere
modell = smf.ols(formel,data=df)
# deretter tilpasse
resultat = modell.fit()

# Steg 4: Presenter resultater fra den tilpassede regresjonsmodellen
print(resultat.summary())    
    
## Tilpass multippel linear regresjon (alle variabler):

formel='bodyfat ~ bmi + age + weight + neck + abdomen + hip'

# Steg 3: Initaliser og tilpass en enkel lineær regresjonsmodell
# først initialisere
modell = smf.ols(formel,data=df)
# deretter tilpasse
resultat = modell.fit()

# Steg 4: Presenter resultater fra den tilpassede regresjonsmodellen
print(resultat.summary())


# Derive intercept b and slope m
# m, b = np.polyfit(df['bmi'],df['bodyfat'], 1)
# sns.relplot(x = 'bmi', y = 'bodyfat', kind = 'scatter', data = df)
# plt.plot(x, m*x + b)
# plt.show()

sns.regplot(df['bmi'],df['bodyfat'])


## Model checking

# kodechunk Steg5

# Steg 5: Evaluere om modellen passer til dataene
# Plotte predikert verdi mot residual 
sns.scatterplot(x=resultat.fittedvalues, y=resultat.resid)
plt.ylabel("Residual")
plt.xlabel("Predikert verdi")
plt.show()

# Lage kvantil-kvantil-plott for residualene
sm.qqplot(resultat.resid,line='45',fit=True)
plt.ylabel("Kvantiler i residualene")
plt.xlabel("Kvantiler i normalfordelingen")
plt.show()














