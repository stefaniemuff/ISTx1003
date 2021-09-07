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
df = pd.read_csv("https://www.math.ntnu.no/emner/IST100x/ISTx1003/Idrett.csv", sep = ',')
df.sort_values(by=['Hoeyde'],inplace=True) # alt sortert på Hoeyde, bare for gøy

# Skriv ut de første og siste radene
print(df)

# Konverter kjønn og idrettsgren til kategory
df=df.astype({'Kjoenn':'category','Sport':'category'})
print(df["Kjoenn"].value_counts())
print(df["Sport"].value_counts())

df.describe()

# Oppgave 1b, enkel linear regresjon

sns.relplot(x = 'Hoeyde', y = 'Blodceller', kind = 'scatter', data = df)
plt.show()

# kodechunk Steg2-4

# Steg 2: spesifiser matematisk modell
# Formula for lin reg:
formel='Blodceller ~ Hoeyde'

# Steg 3: Initaliser og tilpass en enkel lineær regresjonsmodell
# først initialisere
modell = smf.ols(formel,data=df)
# deretter tilpasse
resultat = modell.fit()

# Steg 4: Presenter resultater fra den tilpassede regresjonsmodellen
print(resultat.summary())



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
sns.pairplot(df, vars = ['Blodceller','Hoeyde', 'Vekt'],
             diag_kind = 'kde',
             plot_kws=dict(alpha=0.4))
plt.show()

# Boksplott av Blodceller for hvert Kjoenn og for hver Sport

ax = sns.boxplot(x="Kjoenn", y="Blodceller", data=df)
plt.show()
ax = sns.boxplot(x="Sport", y="Blodceller", data=df)
plt.show()

sns.pairplot(df, vars = ['Hoeyde', 'Vekt', 'Blodceller'],
             hue = 'Kjoenn', 
             diag_kind = 'kde',
             plot_kws=dict(alpha=0.4))
plt.show()

ax = sns.boxplot(x="Sport", y="Blodceller", hue="Kjoenn",
                 data=df, palette="Set3")
plt.show()


## Tilpass multippel linear regresjon:
    
formel='Blodceller ~ Hoeyde + Vekt + Kjoenn + Sport'


modell = smf.ols(formel,data=df)
# deretter tilpasse
resultat = modell.fit()

# Steg 4: Presenter resultater fra den tilpassede regresjonsmodellen
print(resultat.summary())













