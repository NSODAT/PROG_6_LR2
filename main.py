import pandas as pd
import numpy as np
from scipy import stats
import re

# Загрузка данных
train_df = pd.read_csv('train.csv')

# 1. Количество мужчин и женщин
male_count = (train_df['Sex'] == 'male').sum()
female_count = (train_df['Sex'] == 'female').sum()
print(f"1. {male_count} {female_count}")

# 2. Пассажиры по портам посадки
embarked_counts = train_df['Embarked'].value_counts().sort_index()
print(f"2. {embarked_counts['C']} {embarked_counts['Q']} {embarked_counts['S']}")

# 3. Доля погибших
died_count = (train_df['Survived'] == 0).sum()
died_percent = died_count / len(train_df) * 100
print(f"3. {died_count} {died_percent:.2f}%")

# 4. Доли пассажиров по классам
class_dist = train_df['Pclass'].value_counts(normalize=True).sort_index() * 100
print(f"4. {class_dist[1]:.1f}% {class_dist[2]:.1f}% {class_dist[3]:.1f}%")

# 5. Корреляция SibSp-Parch
corr_sibsp_parch, _ = stats.pearsonr(train_df['SibSp'], train_df['Parch'])
print(f"5. {corr_sibsp_parch:.3f}")

# 6. Корреляции с Survived
# Возраст и Survived (с удалением пропусков)
age_surv_corr, _ = stats.pearsonr(train_df['Age'].dropna(), train_df.loc[train_df['Age'].notna(), 'Survived'])
# Пол и Survived
train_df['Sex_numeric'] = train_df['Sex'].map({'male': 0, 'female': 1})
sex_surv_corr, _ = stats.pearsonr(train_df['Sex_numeric'], train_df['Survived'])
# Класс и Survived
pclass_surv_corr, _ = stats.pearsonr(train_df['Pclass'], train_df['Survived'])
print(f"6. {age_surv_corr:.3f} {sex_surv_corr:.3f} {pclass_surv_corr:.3f}")

# 7. Статистики возраста
age_stats = train_df['Age'].agg(['mean', 'median', 'min', 'max'])
print(f"7. Средний: {age_stats['mean']:.1f}, Медиана: {age_stats['median']}, Мин: {age_stats['min']}, Макс: {age_stats['max']}")

# 8. Статистики цены билета
fare_stats = train_df['Fare'].agg(['mean', 'median', 'min', 'max'])
print(f"8. Средняя: {fare_stats['mean']:.2f}, Медиана: {fare_stats['median']:.2f}, Мин: {fare_stats['min']}, Макс: {fare_stats['max']:.2f}")

# 9-10. Анализ имен
def extract_name(name):
    # Ищем первое имя после титула
    match = re.search(r' ([A-Za-z]+)\.', name)
    if match:
        parts = name.split('. ')[1].split()
        return parts[0].strip("()\"'") if len(parts) > 0 else None
    return None

# 9. Самое популярное мужское имя
male_names = train_df[train_df['Sex'] == 'male']['Name'].apply(extract_name).dropna()
print(f"9. {male_names.mode()[0]}")

# 10. Популярные имена среди взрослых
adult_males = train_df[(train_df['Sex'] == 'male') & (train_df['Age'] > 15)]
male_adult_names = adult_males['Name'].apply(extract_name).dropna()

adult_females = train_df[(train_df['Sex'] == 'female') & (train_df['Age'] > 15)]
female_adult_names = adult_females['Name'].apply(extract_name).dropna()

print(f"10. М: {male_adult_names.mode()[0]}, Ж: {female_adult_names.mode()[0]}")