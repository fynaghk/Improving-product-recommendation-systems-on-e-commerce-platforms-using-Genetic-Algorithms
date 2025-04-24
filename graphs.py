import matplotlib.pyplot as plt
import numpy as np

# Simulated fitness values for each user type over 20 generations
generations = np.arange(1, 21)
tech_enthusiast = np.linspace(0.55, 0.85, 20) + np.random.normal(0, 0.01, 20)
bargain_hunter = np.linspace(0.5, 0.78, 20) + np.random.normal(0, 0.01, 20)
critical_reviewer = np.linspace(0.45, 0.71, 20) + np.random.normal(0, 0.01, 20)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(generations, tech_enthusiast, label='Tech Enthusiast')
plt.plot(generations, bargain_hunter, label='Bargain Hunter')
plt.plot(generations, critical_reviewer, label='Critical Reviewer')

plt.title('Şəkil 6. Fərqli istifadəçi növləri üzrə fitness funksiyasının dəyişimi')
plt.xlabel('Nəsil (Generation)')
plt.ylabel('Orta Fitness Dəyəri')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






import matplotlib.pyplot as plt
import numpy as np

# Ortalama reytinq dəyərləri (simulyasiya olunmuş) hər metod üçün
methods = ['GA', 'Top-Rated', 'Random']
avg_ratings = [4.5, 4.2, 3.6]

# Şəkil 7 - Ortalama reytinq müqayisəsi
plt.figure(figsize=(8, 6))
bars = plt.bar(methods, avg_ratings)
plt.title('Şəkil 7. GA, Top-Rated və Random tövsiyə metodlarının ortalama reytinq müqayisəsi')
plt.ylabel('Ortalama Reytinq')
plt.ylim(3, 5)
plt.grid(axis='y')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()






# Kateqoriya müxtəlifliyi üçün diversity ölçüləri (simulyasiya olunmuş)
methods = ['GA', 'Top-Rated', 'Random']
category_diversity = [0.82, 0.65, 0.47]  # Diversity score (0-1 aralığında)

# Şəkil 8 - Tövsiyələrin müxtəliflik səviyyəsi
plt.figure(figsize=(8, 6))
bars = plt.bar(methods, category_diversity, color=['#4CAF50', '#2196F3', '#FF9800'])
plt.title('Şəkil 8. GA, Top-Rated və Random metodlarının tövsiyə müxtəlifliyi müqayisəsi')
plt.ylabel('Müxtəliflik Səviyyəsi (Diversity Score)')
plt.ylim(0, 1)
plt.grid(axis='y')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()



