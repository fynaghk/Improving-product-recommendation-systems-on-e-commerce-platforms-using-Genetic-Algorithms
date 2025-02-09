!pip install dask

!pip install deap

import dask.dataframe as dd

# Google Drive fayl yolunu düz göstərin
file_path = "/content/drive/MyDrive/amazon/amazon_reviews.tsv"

# Dask ilə dataset-i yükləyirik və sütun tiplərini təyin edirik
df = dd.read_csv(file_path, sep="\t", on_bad_lines='skip', 
                 dtype={'helpful_votes': 'float64', 
                        'star_rating': 'float64', 
                        'total_votes': 'float64'})

# İlk 5 sətri göstər
df.head()


import pandas as pd

# Yalnız ilk 500,000 sətri Pandas DataFrame-ə çeviririk (əgər RAM icazə versə, n=1,000,000 edə bilərik)
df = df.compute().sample(n=500000, random_state=42)

print(df.head())  # İlk 5 sətri göstər


from scipy.sparse import csr_matrix

# İstifadəçi və məhsul ID-lərini ədədi formatlara çeviririk
df['customer_id'] = df['customer_id'].astype("category").cat.codes
df['product_id'] = df['product_id'].astype("category").cat.codes

# Sparse matris yaradırıq (RAM dostu)
user_product_sparse = csr_matrix((df['star_rating'], (df['customer_id'], df['product_id'])))

print("Sparse Matris Yaradıldı!", user_product_sparse.shape)



import numpy as np
import random
from deap import base, creator, tools, algorithms

# Genetik alqoritm üçün uyğunluq funksiyası
def fitness(individual):
    recommended_products = [i for i in range(len(individual)) if individual[i] == 1]
    return (sum(user_product_sparse[:, recommended_products].toarray().mean(axis=0)),)  # Reytinqlərin ortalamasını maksimumlaşdırır

# Genetik alqoritm üçün fərd və fitness siniflərini yaradırıq
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox qurulması
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=user_product_sparse.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)






