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






!pip install dask
!pip install deap

from google.colab import drive
drive.mount('/content/drive')


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


from google.colab import drive
drive.mount('/content/drive')




# 1. Lazımi kitabxanaları quraşdır və yüklə
!pip install dask
!pip install deap


import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms

from google.colab import drive
drive.mount('/content/drive')


import dask.dataframe as dd


# 2. Dataseti yüklə (öz yolunu əlavə et əgər Google Drive istifadə edirsənsə)
file_path = "/content/drive/MyDrive/amazon/amazon_reviews.tsv"

# Faylı tab ilə ayrılmış formatda (TSV) oxuyuruq
df = pd.read_csv(file_path, sep="\t", encoding='utf-8', on_bad_lines='skip')


df = df[['product_title', 'star_rating']].dropna()
df['product_title'] = df['product_title'].str.lower()

# 3. Fidanın maraqlarına uyğun açar sözlər
interests = [
    "usb", "hdmi", "logitech", "keyboard", "mouse", 
    "monitor", "tech", "data", "cable", "adapter", 
    "ssd", "external hard drive", "power bank", "charger"
]

# 4. Maraqlı məhsulları seç
mask = df['product_title'].apply(lambda title: any(keyword in title for keyword in interests))
interested_products = df[mask]

# 5. Ortalama reytinqi hesabla
top_products = interested_products.groupby('product_title')['star_rating'].mean().sort_values(ascending=False).head(20)

# 6. Genetik Alqoritm üçün məlumatları hazırlayırıq
product_titles = top_products.index.tolist()
product_scores = top_products.values

# 7. Genetik alqoritmin uyğunluq funksiyası
def fitness(individual):
    selected = [i for i in range(len(individual)) if individual[i] == 1]
    if not selected:
        return (0,)
    avg_score = np.mean([product_scores[i] for i in selected])
    return (avg_score,)

# 8. DEAP struktur qurulması
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(product_titles))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 9. Genetik alqoritmi işə salırıq
population = toolbox.population(n=30)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=25, verbose=True)

# 10. Ən yaxşı fərdi seçirik
best = tools.selBest(population, k=1)[0]
selected_products = [product_titles[i] for i in range(len(best)) if best[i] == 1]

# 11. Nəticəni göstər
print("🔍 Fidan üçün genetik alqoritmlə tövsiyə olunan məhsullar:")
for i, p in enumerate(selected_products, 1):
    print(f"{i}. {p.title()}")




import numpy as np
import random
from deap import base, creator, tools, algorithms

# Məhsul adları və süni reytinqlər
product_titles = [
    "Apple Thunderbolt Adapter",
    "USB Drive 8GB",
    "Aliens Technical Manual",
    "AKG K612 Headphones",
    "Innovator’s Dilemma",
    "The Husband's Secret",
    "Test Big Data",
    "Sony VCT Adapter",
    "Roccat Gaming Mousepad"
]

product_scores = [5.0, 4.5, 3.0, 4.2, 4.7, 3.5, 2.5, 3.4, 4.3]

# Fitness funksiyası
def fitness(individual):
    selected = [i for i in range(len(individual)) if individual[i] == 1]
    if not selected:
        return (0,)
    avg_score = np.mean([product_scores[i] for i in selected])
    return (avg_score,)

# Genetik alqoritm üçün struktur
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(product_titles))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Alqoritmi işə sal
population = toolbox.population(n=30)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=25, verbose=True)

# Ən yaxşı fərdi seç
best = tools.selBest(population, k=1)[0]
selected_products = [product_titles[i] for i in range(len(best)) if best[i] == 1]

# Nəticəni göstər
print("🎯 Naghiyeva Fidan üçün tövsiyə olunan məhsul kombinasiyası:")
for i, product in enumerate(selected_products, 1):
    print(f"{i}. {product}")


