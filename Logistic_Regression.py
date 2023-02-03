from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
import pandas as pd

ames_housing = pd.read_csv("../datasets/ames_housing_no_missing.csv")

target_name = "SalePrice"

data, target = ames_housing.drop(columns=target_name), ames_housing[target_name]

# Nous transformons cela en un problème de classification
# Il s'agira de prédire si une maison coute plus de 200 000$ ou non
target = (target > 200_000).astype(int)

#Nous nous focalisons dans un premier uniquement sur les variables numériques
#Début de la sélection des variables numériques
numerical_columns_selector = selector(dtype_exclude=object)

numerical_columns = numerical_columns_selector(data)

data_numeric=data[numerical_columns]
#Fin de la sélection des variables numériques

#Notre modèle qui est un pipeline, standarisation des données puis, regréssion logistique
model = make_pipeline(StandardScaler(), LogisticRegression())

#Nous effectuons 10 validations croisées
cv_results_num = cross_validate(model, data_numeric, target, cv=10)

#Récupération des scores sur les ensembles de test
scores = cv_results_num["test_score"]

print(
    "La précision moyenne de la validation croisée est: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
)
#Score de 0.925 ± 0.012 :)