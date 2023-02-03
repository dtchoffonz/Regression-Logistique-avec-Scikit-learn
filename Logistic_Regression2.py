from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
import pandas as pd

ames_housing = pd.read_csv("../datasets/AmesHousing.csv")

target_name = "SalePrice"

data, target = ames_housing.drop(columns=target_name), ames_housing[target_name]

# Nous transformons cela en un problème de classification
# Il s'agira de prédire si une maison coute plus de 200 000$ ou non
target = (target > 200_000).astype(int)

#Début de la sélection des variables numériques et catégoriques
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)

#Fin de la sélection des variables numériques et catégoriques

#prétraitement des variables
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

#Notre modèle qui est un pipeline, standarisation des données puis, regréssion logistique
preprocessors = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])

model = make_pipeline(preprocessors, LogisticRegression())

#Nous effectuons 10 validations croisées
cv_results_all = cross_validate(model, data, target, cv=10)

#Récupération des scores sur les ensembles de test
scores = cv_results_all["test_score"]

print(
    "La précision moyenne de la validation croisée est: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
)
#Score de 0.926 ± 0.022 :).Un peu mieux que le modèle qui ne traite que les variables numériques
