import pandas as pd
import numpy as np
import requests
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, fisher_exact
from google.colab import files
import os
from tqdm import tqdm  # <-- ДОБАВЛЕНО

# ==================== 1. ЗАГРУЗКА ФАЙЛА ====================
file_name = "hpo_analysis_results (1).xlsx"
if not os.path.exists(file_name):
    print("📤 Загрузите файл hpo_analysis_results (1).xlsx")
    uploaded = files.upload()
    if uploaded:
        file_name = list(uploaded.keys())[0]
        print(f"✅ Загружен: {file_name}")
    else:
        raise FileNotFoundError("Файл не загружен.")
else:
    print(f"✅ Файл найден: {file_name}")

# ==================== 2. ЗАГРУЗКА hp.obo ====================
obo_file = Path("hp.obo")
if not obo_file.exists():
    print("📥 Скачивание hp.obo...")
    url = "http://purl.obolibrary.org/obo/hp.obo"
    resp = requests.get(url, stream=True)
    with open(obo_file, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ hp.obo загружен")

# ==================== 3. ПАРСИНГ ОНТОЛОГИИ ====================
def build_parents(obo_path):
    parents = {}
    current_id = None
    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[Term]'):
                current_id = None
            elif line.startswith('id: HP:'):
                current_id = line[4:].strip()
            elif line.startswith('is_a: HP:'):
                if current_id is None:
                    continue
                match = re.search(r'HP:\d+', line)
                if match:
                    parent_id = match.group()
                    parents.setdefault(current_id, []).append(parent_id)
    return parents

def get_ancestors(term, parents_cache, visited=None):
    if visited is None:
        visited = set()
    if term in visited:
        return set()
    visited.add(term)
    anc = set()
    for p in parents_cache.get(term, []):
        anc.add(p)
        anc.update(get_ancestors(p, parents_cache, visited))
    return anc

parents = build_parents(obo_file)
print(f"✅ Построены связи для {len(parents)} терминов")

# ==================== 4. ЗАГРУЗКА МАТРИЦЫ И ВЕСОВ ====================
mat = pd.read_excel(file_name, sheet_name="gene_term_matrix_max", index_col=0)
weights = pd.read_excel(file_name, sheet_name="term_weights_tfidf", index_col=0)['0']
all_terms = mat.columns.tolist()
print(f"📊 Матрица: {mat.shape[0]} генов × {mat.shape[1]} терминов")

# ==================== 5. ПОСТРОЕНИЕ ПРЕДКОВ ДЛЯ ВСЕХ ТЕРМИНОВ ====================
ancestor_map = {}
for term in tqdm(all_terms, desc="Построение предков"):
    ancestor_map[term] = get_ancestors(term, parents)
print(f"✅ Предки построены для {len(ancestor_map)} терминов")

# ==================== 6. ОПРЕДЕЛЕНИЕ ОСЕЙ ====================
EXPERT_TERM_TO_AXIS = {
    "HP:0002419": "Neuro_PosteriorFossa", "HP:0001321": "Neuro_PosteriorFossa",
    "HP:0001305": "Neuro_PosteriorFossa", "HP:0002326": "Neuro_PosteriorFossa",
    "HP:0003272": "Neuro_Cortical", "HP:0002089": "Neuro_Cortical",
    "HP:0001274": "Neuro_Cortical", "HP:0001249": "Neuro_Cortical",
    "HP:0001251": "Neuro_Cortical", "HP:0001250": "Neuro_Cortical",
    "HP:0000252": "Neuro_Cortical",
    "HP:0000090": "Renal", "HP:0000083": "Renal", "HP:0000107": "Renal",
    "HP:0000113": "Renal", "HP:0000092": "Renal",
    "HP:0000556": "Ocular", "HP:0000589": "Ocular", "HP:0000648": "Ocular",
    "HP:0000518": "Ocular", "HP:0000568": "Ocular", "HP:0000639": "Ocular",
    "HP:0000486": "Ocular",
    "HP:0001162": "Skeletal", "HP:0000773": "Skeletal", "HP:0000774": "Skeletal",
    "HP:0002650": "Skeletal", "HP:0001156": "Skeletal", "HP:0001363": "Skeletal",
    "HP:0004322": "Skeletal",
    "HP:0001696": "Laterality", "HP:0000853": "Laterality", "HP:0001651": "Laterality",
    "HP:0004306": "Laterality",
    "HP:0001627": "Cardiac", "HP:0001638": "Cardiac",
    "HP:0001392": "Hepatic", "HP:0001397": "Hepatic",
    "HP:0000365": "Hearing",
    "HP:0000175": "Craniofacial", "HP:0000157": "Craniofacial", "HP:0000494": "Craniofacial",
    "HP:0001513": "Endocrine_Metabolic", "HP:0000135": "Endocrine_Metabolic",
    "HP:0000855": "Endocrine_Metabolic", "HP:0002591": "Endocrine_Metabolic",
    "HP:0003119": "Endocrine_Metabolic"
}

AXIS_RULES = {
    'Laterality': {'HP:0001696', 'HP:0000853', 'HP:0001651', 'HP:0004306', 'HP:0001650', 'HP:0011627', 'HP:0000033'},
    'Cardiac': {'HP:0001627', 'HP:0001626', 'HP:0001638', 'HP:0001643', 'HP:0001671', 'HP:0001712'},
    'Neuro_PosteriorFossa': {'HP:0001317', 'HP:0012330', 'HP:0001321', 'HP:0001305', 'HP:0002326', 'HP:0002419', 'HP:0001274'},
    'Neuro_Cortical': {'HP:0000707', 'HP:0001249', 'HP:0001251', 'HP:0001250', 'HP:0000252', 'HP:0001263', 'HP:0001290'},
    'Renal': {'HP:0000077', 'HP:0000090', 'HP:0000107', 'HP:0000113', 'HP:0000083', 'HP:0000092', 'HP:0005563'},
    'Ocular': {'HP:0000478', 'HP:0000556', 'HP:0000589', 'HP:0000648', 'HP:0000518', 'HP:0000568', 'HP:0000639', 'HP:0000486'},
    'Hearing': {'HP:0000365', 'HP:0000598', 'HP:0000407', 'HP:0008510'},
    'Hepatic': {'HP:0001392', 'HP:0004297', 'HP:0006557', 'HP:0001971'},
    'Skeletal': {'HP:0000924', 'HP:0001162', 'HP:0000773', 'HP:0000774', 'HP:0002650', 'HP:0001156', 'HP:0001363', 'HP:0004322'},
    'Craniofacial': {'HP:0000152', 'HP:0000175', 'HP:0000157', 'HP:0000494', 'HP:0000271', 'HP:0000323'},
    'Endocrine_Metabolic': {'HP:0000818', 'HP:0001939', 'HP:0001513', 'HP:0000135', 'HP:0000855', 'HP:0002591', 'HP:0003119'}
}

def get_axis(term):
    if term in EXPERT_TERM_TO_AXIS:
        return EXPERT_TERM_TO_AXIS[term]
    anc = ancestor_map.get(term, set())
    for axis, terms in AXIS_RULES.items():
        if anc.intersection(terms) or term in terms:
            return axis
    return 'Other'

term_to_axis = {}
for term in all_terms:
    term_to_axis[term] = get_axis(term)

axes = sorted(set(term_to_axis.values()) - {'Other'})
print(f"🗺️ Терминов по осям: {sum(1 for a in term_to_axis.values() if a != 'Other')} из {len(term_to_axis)}")
print(f"   Оси: {axes}")

# ==================== 7. ПЕРЕСЧЁТ ОСЕЙ (БЕЗ ПОРОГА) ====================
axis_scores = pd.DataFrame(index=mat.index, columns=axes)
for axis in axes:
    terms = [t for t, a in term_to_axis.items() if a == axis and t in mat.columns]
    if not terms:
        continue
    sub = mat[terms].astype(float)
    w = weights[terms].values
    weighted_sum = (sub.fillna(0) * w).sum(axis=1)
    weight_sum = (sub.notna() * w).sum(axis=1)
    axis_scores[axis] = weighted_sum / weight_sum.replace(0, np.nan)

total_score = axis_scores.sum(axis=1, min_count=1)
pleiotropy = (axis_scores.notna() & (axis_scores > 0)).sum(axis=1)

print(f"📊 Генов с Total_score (не NaN): {total_score.dropna().shape[0]}")

# ==================== 8. КЛАСТЕРИЗАЦИЯ ====================
axis_filled = axis_scores.fillna(0)
scaler = StandardScaler()
scaled = scaler.fit_transform(axis_filled)
k = 9
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled)
axis_scores['Cluster'] = clusters
total_score_df = total_score.to_frame(name='Total_score')
total_score_df['Cluster'] = clusters

# ==================== 9. СОХРАНЕНИЕ НОВЫХ ЛИСТОВ ====================
with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    axis_scores.to_excel(writer, sheet_name='axis_scores_full_ont')
    total_score_df.to_excel(writer, sheet_name='total_score_full_ont')
    pleiotropy.to_excel(writer, sheet_name='pleiotropy_full_ont')
    pd.DataFrame.from_dict(term_to_axis, orient='index', columns=['Axis']).to_excel(writer, sheet_name='term_axis_full_ont')

print("✅ Новые листы сохранены: axis_scores_full_ont, total_score_full_ont, pleiotropy_full_ont, term_axis_full_ont")

# ==================== 10. ВИЗУАЛИЗАЦИИ ====================
# Гистограмма
plt.figure(figsize=(10,6))
sns.histplot(total_score.dropna(), bins=20, kde=True, color='steelblue')
plt.xlabel('Total_score')
plt.ylabel('Количество генов')
plt.title('Распределение Total_score (полная онтология, без порога)')
plt.tight_layout()
plt.savefig('total_score_histogram_full_ont.png', dpi=300)
plt.show()

# Тепловая карта осей (отсортировано по Total_score)
order = total_score.sort_values(ascending=False).index
axis_sorted = axis_scores.loc[order, axes]
plt.figure(figsize=(12,10))
sns.heatmap(axis_sorted, cmap='viridis', cbar_kws={'label': 'Балл по оси'})
plt.title('Тепловая карта осей (гены отсортированы по Total_score)')
plt.tight_layout()
plt.savefig('axis_heatmap_full_ont.png', dpi=300)
plt.show()

# PCA кластеров
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)
plt.figure(figsize=(10,8))
scatter = plt.scatter(pca_result[:,0], pca_result[:,1], c=clusters, cmap='tab10', alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('PCA-проекция генов, окрашенных по кластерам')
plt.colorbar(scatter, label='Кластер')
plt.tight_layout()
plt.savefig('pca_clusters_full_ont.png', dpi=300)
plt.show()

# ==================== 11. ВАЛИДАЦИЯ С ЛЕТАЛЬНОСТЬЮ (если есть) ====================
lethality_file = "lethality.csv"
if os.path.exists(lethality_file):
    leth = pd.read_csv(lethality_file)
    leth['gene'] = leth['gene'].str.upper().str.strip()
    merged = pd.merge(total_score_df, leth, left_index=True, right_on='gene', how='inner')
    if len(merged) > 1:
        cat_map = {
            'prenatal': 4, 'neonatal': 3, 'infantile': 2,
            'childhood': 1, 'juvenile': 1, 'adult': 0, 'young adult': 0,
            'none': 0, 'variable': np.nan
        }
        merged['lethality_score'] = merged['category'].str.lower().map(cat_map)
        merged_valid = merged.dropna(subset=['lethality_score'])
        if len(merged_valid) > 1:
            rho, pval = spearmanr(merged_valid['Total_score'], merged_valid['lethality_score'])
            print(f"📊 Корреляция Спирмена: ρ = {rho:.3f}, p = {pval:.4e} (n={len(merged_valid)})")
            median_score = merged_valid['Total_score'].median()
            merged_valid['high_score'] = merged_valid['Total_score'] > median_score
            merged_valid['lethal'] = merged_valid['lethality_score'] >= 2
            table = pd.crosstab(merged_valid['high_score'], merged_valid['lethal'])
            if table.shape == (2,2):
                oddsratio, p_fisher = fisher_exact(table.values)
                print(f"📊 Точный тест Фишера: OR = {oddsratio:.2f}, p = {p_fisher:.4e}")

# ==================== 12. СКАЧИВАНИЕ ОБНОВЛЁННОГО ФАЙЛА ====================
files.download(file_name)
