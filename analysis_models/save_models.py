import os
import pickle
import pandas as pd

temp = pd.read_pickle("../data/analysis_results.pickle")
best_results = []

for trait in temp['trait'].unique():
    trait_data = temp[temp['trait'] == trait]
    best_idx = trait_data['score'].idxmax()
    best_row = trait_data.loc[best_idx]
    best_results.append(best_row)

best_results = pd.concat(best_results, axis=1).T.reset_index(drop=True)
save_dir = os.path.join(os.getcwd(), "../data", "models")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for index, row in best_results.iterrows():
    trait = row['trait']
    vectorizer = row['vectorizer']
    alg = row['alg']
    filename = f"{trait}_{vectorizer}_{alg}.pickle"
    filepath = os.path.join(save_dir, filename)

    filtered_df = temp[(temp['trait'] == trait) & (temp['vectorizer'] == vectorizer) & (temp['alg'] == alg)]
    print(filtered_df)
    with open(filepath, 'wb') as f:
        pickle.dump(filtered_df, f)

    print(f"Saved {filename}")
