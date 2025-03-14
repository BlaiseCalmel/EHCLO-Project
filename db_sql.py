import psycopg2

try:
    conn = psycopg2.connect(
        dbname="base_test",
        user="bcalmel",
        password="INRAE0range!",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    cur.execute("INSERT INTO utilisateurs (nom, age) VALUES (%s, %s)", ("Alice", 30))

    conn.commit()
    cur.close()
    conn.close()
    print("Donnée insérée avec succès")

except psycopg2.Error as e:
    print("Erreur PostgreSQL :", e)

import xarray as xr

# Charger le fichier NetCDF
dataset = xr.open_dataset('chemin/vers/fichier.nc')

# Voir les dimensions et variables du dataset
print(dataset)

import pandas as pd

# Extraire la variable var1 en fonction de dim1
df_var1 = dataset['var1'].to_dataframe().reset_index()

# Voir les premières lignes de df_var1
print(df_var1.head())

# Extraire var2 et réinitialiser les indices
df_var2 = dataset['var2'].to_dataframe().reset_index()

# Voir les premières lignes de df_var2
print(df_var2.head())


# CREATE TABLE var1_table (
#     id SERIAL PRIMARY KEY,
# dim1 INT,
# var1 DOUBLE PRECISION
# );
# CREATE TABLE var2_table (
#     id SERIAL PRIMARY KEY,
# dim1 INT,
# dim2 INT,
# var2 DOUBLE PRECISION
# );

import psycopg2

# Connexion à PostgreSQL
conn = psycopg2.connect(
    dbname="base_test",
    user="bcalmel",
    password="mon_mot_de_passe",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Insérer les données de var1
for index, row in df_var1.iterrows():
    cur.execute("""
        INSERT INTO var1_table (dim1, var1)
        VALUES (%s, %s)
    """, (row['dim1'], row['var1']))

# Committer les changements
conn.commit()

# Fermer la connexion
cur.close()
conn.close()

# Connexion à PostgreSQL
conn = psycopg2.connect(
    dbname="base_test",
    user="bcalmel",
    password="mon_mot_de_passe",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Insérer les données de var2
for index, row in df_var2.iterrows():
    cur.execute("""
        INSERT INTO var2_table (dim1, dim2, var2)
        VALUES (%s, %s, %s)
    """, (row['dim1'], row['dim2'], row['var2']))

# Committer les changements
conn.commit()

# Fermer la connexion
cur.close()
conn.close()


# Vérifier les données
# SELECT * FROM var1_table;
# SELECT * FROM var2_table;


# Insertion en batch pour aller plus vite
from psycopg2.extras import execute_values

# Préparer les données à insérer (liste des tuples)
data_var1 = df_var1[['dim1', 'var1']].values.tolist()
data_var2 = df_var2[['dim1', 'dim2', 'var2']].values.tolist()

# Connexion à PostgreSQL
conn = psycopg2.connect(
    dbname="base_test",
    user="bcalmel",
    password="mon_mot_de_passe",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Insertion en lot pour var1
query_var1 = "INSERT INTO var1_table (dim1, var1) VALUES %s"
execute_values(cur, query_var1, data_var1)

# Insertion en lot pour var2
query_var2 = "INSERT INTO var2_table (dim1, dim2, var2) VALUES %s"
execute_values(cur, query_var2, data_var2)

# Committer les changements
conn.commit()

# Fermer la connexion
cur.close()
conn.close()
