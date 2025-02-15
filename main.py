!pip install dask
!pip install deap

import dask.dataframe as dd

file_path = "/content/drive/MyDrive/amazon/amazon_reviews.tsv"

# Dask ilə dataset-i yükləyirik və sütun tiplərini təyin edirik
df = dd.read_csv(file_path, sep="\t", on_bad_lines='skip', 
                 dtype={'helpful_votes': 'float64', 
                        'star_rating': 'float64', 
                        'total_votes': 'float64'})

# İlk 5 sətri göstəririk
df.head()


import pandas as pd

# Yalnız ilk 500,000 sətri Pandas DataFrame-ə çeviririk
df = df.compute().sample(n=500000, random_state=42)
print(df.head())

from scipy.sparse import csr_matrix

# İstifadəçi və məhsul ID-lərini ədədi formatlara çeviririk
df['customer_id'] = df['customer_id'].astype("category").cat.codes
df['product_id'] = df['product_id'].astype("category").cat.codes

# Sparse matris yaradırıq
user_product_sparse = csr_matrix((df['star_rating'], (df['customer_id'], df['product_id'])))

print("Sparse Matris Yaradıldı!", user_product_sparse.shape)



import numpy as np
import random
from deap import base, creator, tools, algorithms

# Genetik alqoritm üçün uyğunluq funksiyası
def fitness(individual):
    recommended_products = [i for i in range(len(individual)) if individual[i] == 1]
    # Reytinqlərin ortalamasını maksimumlaşdırır
    return (sum(user_product_sparse[:, recommended_products].toarray().mean(axis=0)),) 
  
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





def run_ga():
    population = toolbox.population(n=50)  # 50 fərqli fərd yaradırıq
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=True)

    # Ən yaxşı fərdi (optimal tövsiyələri) tapırıq
    best_individual = tools.selBest(population, k=1)[0]
    recommended_products = [i for i in range(len(best_individual)) if best_individual[i] == 1]

    return recommended_products

# Genetik Alqoritmi İşlədək
recommended_products = run_ga()

# Tövsiyə olunan məhsulları çap edirik
print("Tövsiyə olunan məhsullar:", recommended_products)







from deap import tools

# Statistik logbook üçün setup
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

logbook = tools.Logbook()
logbook.header = ["gen", "avg", "max"]




pop = toolbox.population(n=50)  # 50 fərdlik başlanğıc populyasiya
hof = tools.HallOfFame(1)  # Ən yaxşı fərdi saxlamaq üçün
NGEN = 20  # 20 nəsil simulyasiya edək
CXPB, MUTPB = 0.5, 0.2  # Çarpazlaşma və mutasiya ehtimalları

for gen in range(NGEN):
    offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
    
    fits = list(map(toolbox.evaluate, offspring))
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit

    pop = toolbox.select(offspring, k=len(pop))
    
    record = stats.compile(pop)  # Statistikaları topla
    logbook.record(gen=gen, **record)  # Logbook-a əlavə et
    print(logbook.stream)  # Konsolda göstərin




import matplotlib.pyplot as plt

gen = logbook.select("gen")
avg_fitness = logbook.select("avg")
max_fitness = logbook.select("max")

plt.figure(figsize=(8, 5))
plt.plot(gen, avg_fitness, label="Orta Fitness", linestyle="dashed", marker="o")
plt.plot(gen, max_fitness, label="Maksimum Fitness", marker="s")
plt.xlabel("Nəsil")
plt.ylabel("Fitness Dəyəri")
plt.title("Genetik Alqoritmin İrəliləyişi")
plt.legend()
plt.grid()
plt.show()
