import pandas as pd
import itertools

data = [{'area': 'a1', 'group': 'g1', 'model': 'm1', 'promo': 'promo1', 'p1': 3, 'p2': 5, 'p3': 1}
      , {'area': 'a1', 'group': 'g1', 'model': 'm2', 'promo': 'promo1', 'p1': 1, 'p2': 3, 'p3': 6}
      , {'area': 'a1', 'group': 'g2', 'model': 'm1', 'promo': 'promo1', 'p1': 3, 'p2': 2, 'p3': 6}
      , {'area': 'a1', 'group': 'g2', 'model': 'm2', 'promo': 'promo1', 'p1': 1, 'p2': 5, 'p3': 9}
      , {'area': 'a1', 'group': 'g3', 'model': 'm1', 'promo': 'promo1', 'p1': 3, 'p2': 2, 'p3': 6}
      , {'area': 'a1', 'group': 'g3', 'model': 'm2', 'promo': 'promo1', 'p1': 1, 'p2': 5, 'p3': 9}
      , {'area': 'a1', 'group': 'g4', 'model': 'm1', 'promo': 'promo1', 'p1': 3, 'p2': 2, 'p3': 6}
      , {'area': 'a1', 'group': 'g4', 'model': 'm2', 'promo': 'promo1', 'p1': 1, 'p2': 5, 'p3': 9}]
df = pd.DataFrame.from_dict(data)

# Get unique group combinations
unique_groups = df['group'].unique()
group_pairs = list(itertools.combinations(unique_groups, 2))

# Create an empty DataFrame to append results to
summary_df = pd.DataFrame()

# Loop over group pairs
for g1, g2 in group_pairs:
    pair_df = df[df['group'].isin([g1, g2])]
    promos = pair_df['promo'].unique()

    for promo in promos:
        promo_df = pair_df[pair_df['promo'] == promo]
        grouped = promo_df.groupby('model')[['p1', 'p2', 'p3']].sum().reset_index()

        # Add metadata columns
        grouped['group'] = f"{g1}_{g2}"
        grouped['promo'] = promo
        grouped['area'] = 'a1'

        # Append to summary_df
        summary_df = pd.concat([summary_df, grouped], ignore_index=True)

# Final result
print(summary_df)
print(pd.concat([df, summary_df]))

# Unique group combinations
unique_groups = df['group'].unique()
group_pairs = list(itertools.combinations(unique_groups, 2))

# Create empty DataFrame for results
summary_df = pd.DataFrame()

# Loop through group pairs
for g1, g2 in group_pairs:
    df_g1 = df[df['group'] == g1]
    df_g2 = df[df['group'] == g2]

    # Filter for shared promos
    common_promos = set(df_g1['promo']).intersection(df_g2['promo'])

    for promo in common_promos:
        g1_filtered = df_g1[df_g1['promo'] == promo].set_index('model')
        g2_filtered = df_g2[df_g2['promo'] == promo].set_index('model')

        # Keep only models that exist in both groups
        common_models = g1_filtered.index.intersection(g2_filtered.index)

        g1_common = g1_filtered.loc[common_models]
        g2_common = g2_filtered.loc[common_models]

        # Apply multipliers
        result = pd.DataFrame(index=common_models)
        result['p1'] = g1_common['p1'] * 2 + g2_common['p1'] * 3
        result['p2'] = g1_common['p2'] * 3 + g2_common['p2'] * 4
        result['p3'] = g1_common['p3'] * -1 + g2_common['p3'] * -2

        # Reset index and add metadata
        result = result.reset_index()
        result['group_pair'] = f"{g1}_{g2}"
        result['promo'] = promo

        # Append to summary
        summary_df = pd.concat([summary_df, result], ignore_index=True)

# Output
print(summary_df)










# Input data
data = [
    {'area': 'a1', 'group': 'g1', 'model': 'm1', 'promo': 'promo1', 'p1': 3, 'p2': 5, 'p3': 1},
    {'area': 'a1', 'group': 'g1', 'model': 'm2', 'promo': 'promo1', 'p1': 1, 'p2': 3, 'p3': 6}
]

df = pd.DataFrame(data)

# Melt to long format for ranking
melted = df.melt(id_vars=['model'], value_vars=['p1', 'p2', 'p3'],
                 var_name='parameter', value_name='value')

# Dense rank with ascending=False (highest value gets rank 1)
# This assigns 1 to the highest value, 2 to second highest, etc.
melted['rank_asc'] = melted['value'].rank(ascending=False, method='dense')

# Now invert rank so highest value gets highest number rank:
max_rank = melted['rank_asc'].max()
melted['rank'] = max_rank - melted['rank_asc'] + 1

# Pivot ranks back to wide format
rank_df = melted.pivot(index='model', columns='parameter', values='rank').add_suffix('_rank').reset_index()

# Merge ranks into original dataframe
final_df = df.merge(rank_df, on='model')
print(final_df)
print(final_df[['model', 'p1', 'p1_rank', 'p2', 'p2_rank', 'p3', 'p3_rank']])












import pandas as pd

# Input data
data = [
    {'area': 'a1', 'group': 'g1', 'model': 'm1', 'promo': 'promo1', 'p1': 3, 'p2': 5, 'p3': 1},
    {'area': 'a1', 'group': 'g1', 'model': 'm2', 'promo': 'promo1', 'p1': 1, 'p2': 3, 'p3': 6},
    {'area': 'a1', 'group': 'g2', 'model': 'm1', 'promo': 'promo1', 'p1': 3, 'p2': 2, 'p3': 6},
    {'area': 'a1', 'group': 'g2', 'model': 'm2', 'promo': 'promo1', 'p1': 1, 'p2': 5, 'p3': 9},
    {'area': 'a1', 'group': 'g3', 'model': 'm1', 'promo': 'promo1', 'p1': 3, 'p2': 2, 'p3': 6},
    {'area': 'a1', 'group': 'g3', 'model': 'm2', 'promo': 'promo1', 'p1': 1, 'p2': 5, 'p3': 9},
    {'area': 'a1', 'group': 'g4', 'model': 'm1', 'promo': 'promo1', 'p1': 3, 'p2': 2, 'p3': 6},
    {'area': 'a1', 'group': 'g4', 'model': 'm2', 'promo': 'promo1', 'p1': 1, 'p2': 5, 'p3': 9}
]
df = pd.DataFrame(data)

# Melt to long format for flexible ranking
melted = df.melt(id_vars=['area', 'group', 'model', 'promo'], 
                 value_vars=['p1', 'p2', 'p3'], 
                 var_name='parameter', value_name='value')

# Group by group+promo and rank across all p1/p2/p3 values in that group
melted['rank'] = (
    melted
    .groupby(['group', 'promo'])['value']
    .rank(ascending=False, method='dense')
)

# Invert so higher values get higher rank numbers
melted['rank'] = (
    melted.groupby(['group', 'promo'])['rank']
    .transform(lambda x: x.max() - x + 1)
)

# Pivot rank columns back to wide format
rank_df = melted.pivot_table(index=['area', 'group', 'model', 'promo'], 
                              columns='parameter', 
                              values='rank').add_suffix('_rank').reset_index()

# Merge rank columns into original dataframe
final_df = df.merge(rank_df, on=['area', 'group', 'model', 'promo'])

# Reorder for readability
final_df = final_df[['area', 'group', 'model', 'promo', 
                     'p1', 'p1_rank', 'p2', 'p2_rank', 'p3', 'p3_rank']]

# Show final result
print(final_df)









import pandas as pd

result_data = [{'prod_group': 'bike', 'prod': 'bike1', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike1', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor1', 'seller': 'us', 'promo': 'promo1/promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor3', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor4', 'seller': 'eu', 'promo': 'promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}]
priority_data = [{'prod_group': 'bike', 'seller': 'us', 'promo': 'promo1', 'pp': 1, 'p1': 11, 'p2': 12, 'p3': 13}, {'prod_group': 'bike', 'seller': 'eu', 'promo': 'promo1', 'pp': 2, 'p1': 22, 'p2': 23, 'p3': 24}, {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo1', 'pp': 3, 'p1': 33, 'p2': 34, 'p3': 35}, {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo2', 'pp': 4, 'p1': 44, 'p2': 45, 'p3': 46}, {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo1', 'pp': 5, 'p1': 55, 'p2': 56, 'p3': 57}, {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo2', 'pp': 6, 'p1': 66, 'p2': 67, 'p3': 68}]

# Create DataFrames
df_result = pd.DataFrame(result_data)
df_priority = pd.DataFrame(priority_data)

# Merge on prod_group, seller, and promo
df_updated = df_result.drop(columns=['pp', 'p1', 'p2', 'p3']).merge(
    df_priority,
    on=['prod_group', 'seller', 'promo'],
    how='left'
)

# Fill any unmatched values with 0 and ensure integer type
df_updated[['pp', 'p1', 'p2', 'p3']] = df_updated[['pp', 'p1', 'p2', 'p3']].fillna(0).astype(int)

# Show result
print(df_updated)




import pandas as pd

result_data = [{'prod_group': 'bike', 'prod': 'bike1', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike1', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor1', 'seller': 'us', 'promo': 'promo1/promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor3', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor4', 'seller': 'eu', 'promo': 'promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}]
priority_data = [{'prod_group': 'bike', 'seller': 'us', 'promo': 'promo1', 'pp': 1, 'p1': 11, 'p2': 12, 'p3': 13}, {'prod_group': 'bike', 'seller': 'eu', 'promo': 'promo1', 'pp': 2, 'p1': 22, 'p2': 23, 'p3': 24}, {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo1', 'pp': 3, 'p1': 33, 'p2': 34, 'p3': 35}, {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo2', 'pp': 4, 'p1': 44, 'p2': 45, 'p3': 46}, {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo1', 'pp': 5, 'p1': 55, 'p2': 56, 'p3': 57}, {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo2', 'pp': 6, 'p1': 66, 'p2': 67, 'p3': 68}]

# Create DataFrames
df_result = pd.DataFrame(result_data)
df_priority = pd.DataFrame(priority_data)


# Function to look up and sum values for possibly multiple promos
def get_priority_values(row):
    promo_list = row['promo'].split('/')
    values = {'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}

    for promo in promo_list:
        match = df_priority[
            (df_priority['prod_group'] == row['prod_group']) &
            (df_priority['seller'] == row['seller']) &
            (df_priority['promo'] == promo)
            ]
        if not match.empty:
            values['pp'] += match['pp'].iloc[0]
            values['p1'] += match['p1'].iloc[0]
            values['p2'] += match['p2'].iloc[0]
            values['p3'] += match['p3'].iloc[0]
    return pd.Series(values)


# Apply function to each row
df_result[['pp', 'p1', 'p2', 'p3']] = df_result.apply(get_priority_values, axis=1)

# Result
print(df_result)



















import pandas as pd

result_data = [{'prod_group': 'bike', 'prod': 'bike1', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike1', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor1', 'seller': 'us', 'promo': 'promo1/promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor3', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor4', 'seller': 'eu', 'promo': 'promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}]
priority_data = [{'prod_group': 'bike', 'seller': 'us', 'promo': 'promo1', 'pp': 1, 'p1': 11, 'p2': 12, 'p3': 13}, {'prod_group': 'bike', 'seller': 'eu', 'promo': 'promo1', 'pp': 2, 'p1': 22, 'p2': 23, 'p3': 24}, {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo1', 'pp': 3, 'p1': 33, 'p2': 34, 'p3': 35}, {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo2', 'pp': 4, 'p1': 44, 'p2': 45, 'p3': 46}, {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo1', 'pp': 5, 'p1': 55, 'p2': 56, 'p3': 57}, {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo2', 'pp': 6, 'p1': 66, 'p2': 67, 'p3': 68}]

# Create DataFrames
df_result = pd.DataFrame(result_data)
df_priority = pd.DataFrame(priority_data)

# ---- Step 1: Handle rows with only one promo ----
single_promo_mask = ~df_result['promo'].str.contains('/')
df_result.loc[single_promo_mask, ['pp', 'p1', 'p2', 'p3']] = (
    df_result[single_promo_mask]
    .merge(df_priority, on=['prod_group', 'seller', 'promo'], how='left')[['pp_y', 'p1_y', 'p2_y', 'p3_y']]
    .fillna(0)
    .astype(int)
    .rename(columns={'pp_y': 'pp', 'p1_y': 'p1', 'p2_y': 'p2', 'p3_y': 'p3'})
)

# ---- Step 2: Handle rows with multiple promos ("/") ----
def get_multi_promo_sum(row):
    promo_list = row['promo'].split('/')
    total = {'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}
    for promo in promo_list:
        match = df_priority[
            (df_priority['prod_group'] == row['prod_group']) &
            (df_priority['seller'] == row['seller']) &
            (df_priority['promo'] == promo)
        ]
        if not match.empty:
            total['pp'] += match['pp'].iloc[0]
            total['p1'] += match['p1'].iloc[0]
            total['p2'] += match['p2'].iloc[0]
            total['p3'] += match['p3'].iloc[0]
    return pd.Series(total)

multi_promo_mask = df_result['promo'].str.contains('/')
df_result.loc[multi_promo_mask, ['pp', 'p1', 'p2', 'p3']] = (
    df_result[multi_promo_mask].apply(get_multi_promo_sum, axis=1)
)

# ---- Final Result ----
print(df_result)















import pandas as pd

result_data = [{'prod_group': 'bike', 'prod': 'bike1', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike1', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor1', 'seller': 'us', 'promo': 'promo1/promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor3', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor4', 'seller': 'eu', 'promo': 'promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}]
priority_data = [{'prod_group': 'bike', 'seller': 'us', 'promo': 'promo1', 'pp': 1, 'p1': 11, 'p2': 12, 'p3': 13}, {'prod_group': 'bike', 'seller': 'eu', 'promo': 'promo1', 'pp': 2, 'p1': 22, 'p2': 23, 'p3': 24}, {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo1', 'pp': 3, 'p1': 33, 'p2': 34, 'p3': 35}, {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo2', 'pp': 4, 'p1': 44, 'p2': 45, 'p3': 46}, {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo1', 'pp': 5, 'p1': 55, 'p2': 56, 'p3': 57}, {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo2', 'pp': 6, 'p1': 66, 'p2': 67, 'p3': 68}]

# Create DataFrames
df_result = pd.DataFrame(result_data)
df_priority = pd.DataFrame(priority_data)

# ---- Step 1: Handle rows with only one promo ----
single_promo_mask = ~df_result['promo'].str.contains('/')
df_result.loc[single_promo_mask, ['pp', 'p1', 'p2', 'p3']] = (
    df_result[single_promo_mask]
    .merge(df_priority, on=['prod_group', 'seller', 'promo'], how='left')[['pp_y', 'p1_y', 'p2_y', 'p3_y']]
    .fillna(0)
    .astype(int)
    .rename(columns={'pp_y': 'pp', 'p1_y': 'p1', 'p2_y': 'p2', 'p3_y': 'p3'})
)

# ---- Step 2: Handle rows with multiple promos ("/") ----
def get_multi_promo_weighted_sum(row):
    promo_list = row['promo'].split('/')
    weights = [1.5, 2.0]  # left promo gets 1.5, right promo gets 2.0
    total = {'pp': 0.0, 'p1': 0.0, 'p2': 0.0, 'p3': 0.0}

    for promo, weight in zip(promo_list, weights):
        match = df_priority[
            (df_priority['prod_group'] == row['prod_group']) &
            (df_priority['seller'] == row['seller']) &
            (df_priority['promo'] == promo)
        ]
        if not match.empty:
            total['pp'] += match['pp'].iloc[0] * weight
            total['p1'] += match['p1'].iloc[0] * weight
            total['p2'] += match['p2'].iloc[0] * weight
            total['p3'] += match['p3'].iloc[0] * weight

    return pd.Series({k: int(round(v)) for k, v in total.items()})

# Apply to multi promo rows
multi_promo_mask = df_result['promo'].str.contains('/')
df_result.loc[multi_promo_mask, ['pp', 'p1', 'p2', 'p3']] = (
    df_result[multi_promo_mask].apply(get_multi_promo_weighted_sum, axis=1)
)

# ---- Final Result ----
print(df_result)











import pandas as pd

# --- Data ---
result_data = [
    {'prod_group': 'bike', 'prod': 'bike1', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'bike', 'prod': 'bike1', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor1', 'seller': 'us', 'promo': 'promo1/promo2/promo3', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor2', 'seller': 'us', 'promo': 'promo1/promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor3', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor4', 'seller': 'eu', 'promo': 'promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}
]

priority_data = [
    {'prod_group': 'bike', 'seller': 'us', 'promo': 'promo1', 'pp': 1, 'p1': 11, 'p2': 12, 'p3': 13},
    {'prod_group': 'bike', 'seller': 'eu', 'promo': 'promo1', 'pp': 2, 'p1': 22, 'p2': 23, 'p3': 24},
    {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo1', 'pp': 3, 'p1': 33, 'p2': 34, 'p3': 35},
    {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo2', 'pp': 4, 'p1': 44, 'p2': 45, 'p3': 46},
    {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo1', 'pp': 5, 'p1': 55, 'p2': 56, 'p3': 57},
    {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo2', 'pp': 6, 'p1': 66, 'p2': 67, 'p3': 68},
    {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo3', 'pp': 7, 'p1': 77, 'p2': 78, 'p3': 79},
]

df_result = pd.DataFrame(result_data)
df_priority = pd.DataFrame(priority_data)

df_result['promo_count'] = df_result['promo'].str.count('/')
df_result = df_result.sort_values('promo_count')  # from 0 → 1 → 2
print(df_result)

# ---- Step 1: Single promo ----
single_promo_mask = ~df_result['promo'].str.contains('/')
df_result.loc[single_promo_mask, ['pp', 'p1', 'p2', 'p3']] = (
    df_result[single_promo_mask]
    .merge(df_priority, on=['prod_group', 'seller', 'promo'], how='left')[['pp_y', 'p1_y', 'p2_y', 'p3_y']]
    .fillna(0)
    .astype(int)
    .rename(columns={'pp_y': 'pp', 'p1_y': 'p1', 'p2_y': 'p2', 'p3_y': 'p3'})
)

# ---- Step 2: Multi-promo with weight ----
def get_weighted_combo_sum(row):
    promos = row['promo'].split('/')
    total = {'pp': 0.0, 'p1': 0.0, 'p2': 0.0, 'p3': 0.0}

    pairs = []
    if len(promos) == 2:
        pairs = [(promos[0], promos[1], 1.5)]
    elif len(promos) == 3:
        pairs = [(promos[0], promos[1], 1.5), (promos[1], promos[2], 2.0)]
    else:
        return pd.Series(total)  # return zeros

    for p1, p2, weight in pairs:
        match1 = df_priority[
            (df_priority['prod_group'] == row['prod_group']) &
            (df_priority['seller'] == row['seller']) &
            (df_priority['promo'] == p1)

        ]

        match2 = df_priority[
            (df_priority['prod_group'] == row['prod_group']) &
            (df_priority['seller'] == row['seller']) &
            (df_priority['promo'] == p2)
        ]
        if not match1.empty and not match2.empty:
            total['pp'] += (match1.iloc[0]['pp'] + match2.iloc[0]['pp']) * weight
            total['p1'] += (match1.iloc[0]['p1'] + match2.iloc[0]['p1']) * weight
            total['p2'] += (match1.iloc[0]['p2'] + match2.iloc[0]['p2']) * weight
            total['p3'] += (match1.iloc[0]['p3'] + match2.iloc[0]['p3']) * weight

    return pd.Series({k: int(round(v)) for k, v in total.items()})

multi_promo_mask = df_result['promo'].str.contains('/')
df_result.loc[multi_promo_mask, ['pp', 'p1', 'p2', 'p3']] = (
    df_result[multi_promo_mask].apply(get_weighted_combo_sum, axis=1)
)

df_result = df_result.sort_index()
# ---- Final ----
print(df_result)









import pandas as pd

# Sample data
result_data = [
    {'prod_group': 'bike', 'prod': 'bike1', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'bike', 'prod': 'bike1', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor1', 'seller': 'us', 'promo': 'promo1/promo2/promo3', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor2', 'seller': 'us', 'promo': 'promo1/promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor3', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor4', 'seller': 'eu', 'promo': 'promo2/promo1/promo2/promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
]

priority_data = [
    {'prod_group': 'bike', 'seller': 'us', 'promo': 'promo1', 'pp': 1, 'p1': 11, 'p2': 12, 'p3': 13},
    {'prod_group': 'bike', 'seller': 'eu', 'promo': 'promo1', 'pp': 2, 'p1': 22, 'p2': 23, 'p3': 24},
    {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo1', 'pp': 3, 'p1': 33, 'p2': 34, 'p3': 35},
    {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo2', 'pp': 4, 'p1': 44, 'p2': 45, 'p3': 46},
    {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo1', 'pp': 5, 'p1': 55, 'p2': 56, 'p3': 57},
    {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo2', 'pp': 6, 'p1': 66, 'p2': 67, 'p3': 68},
]

# DataFrames
df_result = pd.DataFrame(result_data)
df_priority = pd.DataFrame(priority_data)

# Save original order
df_result['original_index'] = df_result.index

# Count number of slashes to determine promo complexity
df_result['promo_count'] = df_result['promo'].str.count('/')

# Sort to process simple promos first
df_result = df_result.sort_values('promo_count').reset_index(drop=True)

# --- Step 1: Handle single promo ---
single_mask = df_result['promo_count'] == 0
df_result.loc[single_mask, ['pp', 'p1', 'p2', 'p3']] = (
    df_result[single_mask]
    .merge(df_priority, on=['prod_group', 'seller', 'promo'], how='left')[['pp_y', 'p1_y', 'p2_y', 'p3_y']]
    .fillna(0)
    .astype(int)
    .rename(columns={'pp_y': 'pp', 'p1_y': 'p1', 'p2_y': 'p2', 'p3_y': 'p3'})
)

# --- Step 2: Handle multiple promos ---
def get_weighted_combo_sum(row):
    promos = row['promo'].split('/')
    total = {'pp': 0.0, 'p1': 0.0, 'p2': 0.0, 'p3': 0.0}

    for i in range(len(promos) - 1):
        p1, p2 = promos[i], promos[i + 1]
        weight = 1.5 + i * 0.5  # 1.5, 2.0, 2.5, ...

        match1 = df_priority[
            (df_priority['prod_group'] == row['prod_group']) &
            (df_priority['seller'] == row['seller']) &
            (df_priority['promo'] == p1)
        ]
        match2 = df_priority[
            (df_priority['prod_group'] == row['prod_group']) &
            (df_priority['seller'] == row['seller']) &
            (df_priority['promo'] == p2)
        ]
        if not match1.empty and not match2.empty:
            total['pp'] += (match1['pp'].iloc[0] + match2['pp'].iloc[0]) * weight
            total['p1'] += (match1['p1'].iloc[0] + match2['p1'].iloc[0]) * weight
            total['p2'] += (match1['p2'].iloc[0] + match2['p2'].iloc[0]) * weight
            total['p3'] += (match1['p3'].iloc[0] + match2['p3'].iloc[0]) * weight

    return pd.Series({k: int(round(v)) for k, v in total.items()})

multi_mask = df_result['promo_count'] > 0
df_result.loc[multi_mask, ['pp', 'p1', 'p2', 'p3']] = (
    df_result[multi_mask].apply(get_weighted_combo_sum, axis=1)
)

# Restore original order
df_result = df_result.sort_values('original_index').drop(columns=['original_index', 'promo_count']).reset_index(drop=True)

# --- Final Result ---
print(df_result)













import pandas as pd

# Sample data
result_data = [{'prod_group': 'bike', 'prod': 'bike1', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike1', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor1', 'seller': 'us', 'promo': 'promo1/promo2/promo3', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor2', 'seller': 'us', 'promo': 'promo1/promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor3', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor4', 'seller': 'eu', 'promo': 'promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}, {'prod_group': 'motor', 'prod': 'motor2', 'seller': 'eu', 'promo': 'promo1/promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}]

priority_data = [
    {'prod_group': 'bike', 'seller': 'us', 'promo': 'promo1', 'pp': 1, 'p1': 11, 'p2': 12, 'p3': 13},
    {'prod_group': 'bike', 'seller': 'eu', 'promo': 'promo1', 'pp': 2, 'p1': 22, 'p2': 23, 'p3': 24},
    {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo1', 'pp': 3, 'p1': 33, 'p2': 34, 'p3': 35},
    {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo2', 'pp': 4, 'p1': 44, 'p2': 45, 'p3': 46},
    {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo1', 'pp': 5, 'p1': 55, 'p2': 56, 'p3': 57},
    {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo2', 'pp': 6, 'p1': 66, 'p2': 67, 'p3': 68},
]

# DataFrames
df_result = pd.DataFrame(result_data)
df_priority = pd.DataFrame(priority_data)

# Save original order
df_result['original_index'] = df_result.index

# Count number of slashes to determine promo complexity
df_result['promo_count'] = df_result['promo'].str.count('/')

# Sort to process simple promos first
df_result = df_result.sort_values('promo_count').reset_index(drop=True)

# --- Step 1: Handle single promo ---
single_mask = df_result['promo_count'] == 0
df_result.loc[single_mask, ['pp', 'p1', 'p2', 'p3']] = (
    df_result[single_mask]
    .merge(df_priority, on=['prod_group', 'seller', 'promo'], how='left')[['pp_y', 'p1_y', 'p2_y', 'p3_y']]
    .fillna(0)
    .astype(int)
    .rename(columns={'pp_y': 'pp', 'p1_y': 'p1', 'p2_y': 'p2', 'p3_y': 'p3'})
)

# --- Step 2: Handle multiple promos ---
def get_weighted_combo_sum(row):
    promos = row['promo'].split('/')
    total = {'pp': 0.0, 'p1': 0.0, 'p2': 0.0, 'p3': 0.0}

    for i in range(len(promos) - 1):
        p1, p2 = promos[i], promos[i + 1]
        weight = 1.5 + i * 0.5  # 1.5, 2.0, 2.5, ...

        match1 = df_priority[
            (df_priority['prod_group'] == row['prod_group']) &
            (df_priority['seller'] == row['seller']) &
            (df_priority['promo'] == p1)
        ]
        match2 = df_priority[
            (df_priority['prod_group'] == row['prod_group']) &
            (df_priority['seller'] == row['seller']) &
            (df_priority['promo'] == p2)
        ]
        if not match1.empty and not match2.empty:
            total['pp'] += (match1['pp'].iloc[0] + match2['pp'].iloc[0]) * weight
            total['p1'] += (match1['p1'].iloc[0] + match2['p1'].iloc[0]) * weight
            total['p2'] += (match1['p2'].iloc[0] + match2['p2'].iloc[0]) * weight
            total['p3'] += (match1['p3'].iloc[0] + match2['p3'].iloc[0]) * weight

    return pd.Series({k: int(round(v)) for k, v in total.items()})

multi_mask = df_result['promo_count'] > 0
df_result.loc[multi_mask, ['pp', 'p1', 'p2', 'p3']] = (
    df_result[multi_mask].apply(get_weighted_combo_sum, axis=1)
)

# --- Step 3: Add dense ranks within (prod_group, prod, promo) ---
# Melt the relevant columns for easier ranking
melted = df_result.melt(
    id_vars=['prod_group', 'prod', 'promo', 'seller'],
    value_vars=['p1', 'p2', 'p3'],
    var_name='metric',
    value_name='value'
)

# Add dense rank for each metric within each (prod_group, prod, promo) group
melted['rank'] = (
    melted
    .groupby(['prod_group', 'prod', 'promo', 'metric'])['value']
    .rank(method='dense', ascending=True)
    .astype(int)
)

# Pivot back to wide format
rank_df = (
    melted
    .pivot(index=['prod_group', 'prod', 'seller', 'promo'], columns='metric', values='rank')
    .reset_index()
    .rename(columns={'p1': 'rank_p1', 'p2': 'rank_p2', 'p3': 'rank_p3'})
)

# Merge the ranks into df_result
df_result = df_result.merge(rank_df, on=['prod_group', 'prod', 'seller', 'promo'], how='left')

# Restore original order
df_result = df_result.sort_values('original_index').drop(columns=['original_index', 'promo_count']).reset_index(drop=True)

# --- Final Result ---
print(df_result)


















import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)


result_data = [
    {'prod_group': 'bike', 'prod': 'bike1', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'bike', 'prod': 'bike1', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'us', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'bike', 'prod': 'bike2', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor1', 'seller': 'us', 'promo': 'promo1/promo2/promo3', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor2', 'seller': 'us', 'promo': 'promo1/promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor3', 'seller': 'eu', 'promo': 'promo1', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor4', 'seller': 'eu', 'promo': 'promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0},
    {'prod_group': 'motor', 'prod': 'motor2', 'seller': 'eu', 'promo': 'promo1/promo2', 'pp': 0, 'p1': 0, 'p2': 0, 'p3': 0}
]

priority_data = [
    {'prod_group': 'bike', 'seller': 'us', 'promo': 'promo1', 'pp': 1, 'p1': 11, 'p2': 12, 'p3': 13},
    {'prod_group': 'bike', 'seller': 'eu', 'promo': 'promo1', 'pp': 2, 'p1': 22, 'p2': 23, 'p3': 24},
    {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo1', 'pp': 3, 'p1': 33, 'p2': 34, 'p3': 35},
    {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo2', 'pp': 4, 'p1': 44, 'p2': 45, 'p3': 46},
    {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo1', 'pp': 5, 'p1': 55, 'p2': 56, 'p3': 57},
    {'prod_group': 'motor', 'seller': 'eu', 'promo': 'promo2', 'pp': 6, 'p1': 66, 'p2': 67, 'p3': 68},
    {'prod_group': 'motor', 'seller': 'us', 'promo': 'promo3', 'pp': 1, 'p1': 10, 'p2': 10, 'p3': 10},  # for promo3
]

df_result = pd.DataFrame(result_data)
df_priority = pd.DataFrame(priority_data)

# Define weights per promo
promo_weights = {
    'promo1': 2,
    'promo2': 3,
    'promo3': 4
}

df_result['original_index'] = df_result.index

def weighted_sum(row):
    promos = row['promo'].split('/')
    total = {'pp':0, 'p1':0, 'p2':0, 'p3':0}
    for promo in promos:
        weight = promo_weights.get(promo, 1)
        match = df_priority[
            (df_priority['prod_group'] == row['prod_group']) &
            (df_priority['seller'] == row['seller']) &
            (df_priority['promo'] == promo)
        ]
        if not match.empty:
            total['pp'] += match['pp'].iloc[0] * weight
            total['p1'] += match['p1'].iloc[0] * weight
            total['p2'] += match['p2'].iloc[0] * weight
            total['p3'] += match['p3'].iloc[0] * weight
    return pd.Series(total)

df_result[['pp', 'p1', 'p2', 'p3']] = df_result.apply(weighted_sum, axis=1)

# Rank p1, p2, p3 grouped by exact promo string
df_result['rank_p1'] = df_result.groupby('promo')['p1'].rank(method='dense', ascending=True).astype(int)
df_result['rank_p2'] = df_result.groupby('promo')['p2'].rank(method='dense', ascending=True).astype(int)
df_result['rank_p3'] = df_result.groupby('promo')['p3'].rank(method='dense', ascending=True).astype(int)

# Restore original order
df_result = df_result.sort_values('original_index').reset_index(drop=True)

# Drop helper column
df_result = df_result.drop(columns=['original_index'])

print(df_result)



















import numpy as np

dict_map = {}

main_group = ['m1', 'm2']
sub_group = ['g1', 'g2', 'g3']
random_integers = np.random.randint(1, 10, size=(6))

i = 0
for main in main_group:
    dict_map[main] = {}
    for sub in sub_group:
        level = (random_integers[i] + 0.1) ** 2
        if sub == 'g1':
            grade = 2
        elif sub == 'g2':
            grade = 3
        elif sub == 'g3':
            grade = 4

        dict_map[main][sub] = level * 2
    i += 1

print(dict_map)



main_group = ['m1', 'm2']
sub_group = ['g1', 'g2', 'g3']

# Predefined grades
grade_map = {'g1': 2, 'g2': 3, 'g3': 4}

# Create levels using NumPy
i_array = np.arange(1, len(main_group) + 1)  # [1, 2]
levels = (i_array + 0.1) ** 2                # [1.21, 4.41]
scaled_levels = levels * 2                   # [2.42, 8.82]

# Build dict_map
# dict_map = {
#     main: {
#         sub: scaled_levels[i]
#         for sub in sub_group
#     }
#     for i, main in enumerate(main_group)
# }

dict_map = {
    main: {
        sub: {
            'level': scaled_levels[i],
            'grade': grade_map[sub]
        }
        for sub in sub_group
    }
    for i, main in enumerate(main_group)
}

print(dict_map)

random_integers = np.random.randint(1, 10, size=(6))
print(random_integers[2])





import numpy as np

dict_map = {}

main_group = ['m1', 'm2']
sub_group = ['g1', 'g2', 'g3']
random_integers = np.random.randint(1, 10, size=(6))

i = 0
for main in main_group:
    dict_map[main] = {}
    for sub in sub_group:
        level = (random_integers[i] + 0.1) ** 2
        if sub == 'g1':
            grade = 2
        elif sub == 'g2':
            grade = 3
        elif sub == 'g3':
            grade = 4

        dict_map[main][sub] = level * 2
    i += 1

print(dict_map)



import numpy as np

main_group = ['m1', 'm2']
sub_group = ['g1', 'g2', 'g3']
grade_map = {'g1': 2, 'g2': 3, 'g3': 4}

# Generate random integers (flattened into 2D shape: [main, sub])
random_values = np.random.randint(1, 10, size=(len(main_group), len(sub_group)))
levels = (random_values + 0.1) ** 2
scaled_levels = levels * 2

# Construct dict_map using nested dictionary comprehension
dict_map = {
    main: {
        sub: scaled_levels[i, j]
        for j, sub in enumerate(sub_group)
    }
    for i, main in enumerate(main_group)
}

print(dict_map)




import numpy as np

dict_map = {'m1':
                {'g1': 3
                , 'g2': 4
                , 'g3': 5}
            , 'm2':
                {'g1': 11
                , 'g2': 12
                , 'g3': 13}}

main_group = ['m1', 'm2']
sub_group = ['g1', 'g2', 'g3']


i = 0
for main in main_group:
    for sub in sub_group:
        level = (dict_map[main][sub] + 0.1) ** 2
        if sub == 'g1':
            grade = 2
        elif sub == 'g2':
            grade = 3
        elif sub == 'g3':
            grade = 4

        dict_map[main][sub] = level * 2
    i += 1

print(dict_map)


dict_map = {
    'm1': {'g1': 3, 'g2': 4, 'g3': 5},
    'm2': {'g1': 11, 'g2': 12, 'g3': 13}
}

main_group = ['m1', 'm2']
sub_group = ['g1', 'g2', 'g3']

# Extract values into a NumPy array
data = np.array([[dict_map[main][sub] for sub in sub_group] for main in main_group], dtype=float)

# Apply the transformation using NumPy
scaled = ((data + 0.1) ** 1.2) * 2

# Reconstruct dict_map with updated values
dict_map = {
    main: {
        sub: scaled[i, j]
        for j, sub in enumerate(sub_group)
    }
    for i, main in enumerate(main_group)
}

print(dict_map)



import numpy as np
import pandas as pd

dict_map = {
    'm1': {'g1': 33, 'g2': 24, 'g3': 14},
    'm2': {'g1': 11, 'g2': 12, 'g3': 14}
}

main_group = ['m1', 'm2']
sub_group = ['g1', 'g2', 'g3']

# Step 1: Extract values into NumPy array
data = np.array([[dict_map[main][sub] for sub in sub_group] for main in main_group], dtype=float)
print(data)
# Step 2: Apply the transformation
scaled = ((data + 0.1) ** 1.2) * 2

# Step 3: Flatten and rank using pandas (dense rank)
flat_scaled = scaled.flatten()
ranks = pd.Series(flat_scaled).rank(method='dense').astype(int).values

# Step 4: Reshape ranks back to 2D array
rank_array = ranks.reshape(scaled.shape)

# Step 5: Rebuild dict_map with ranks instead of scaled values
dict_map_ranked = {
    main: {
        sub: rank_array[i, j]
        for j, sub in enumerate(sub_group)
    }
    for i, main in enumerate(main_group)
}

print(dict_map_ranked)





import numpy as np

dict_map = {'m1':
                {'g1': 3
                , 'g2': 4
                , 'g3': 5}
            , 'm2':
                {'g1': 11
                , 'g2': 12
                , 'g3': 13}}

main_group = ['m1', 'm2']
sub_group = ['g1', 'g2', 'g3']



for main in main_group:
    for sub in sub_group:
        level = (dict_map[main][sub] + 0.1) ** 1.5
        if sub == 'g1':
            grade = 2
        elif sub == 'g2':
            grade = 3
        elif sub == 'g3':
            grade = 4

        dict_map[main][sub] = level * grade

print(dict_map)

# Can I improve my code to be more efficient using numpy?
# and then,
# I want to generate a rank for all values and replace the values. If there are values with the same rank, for example, if the 2nd rank value is the same, the next rank should be 3, not 4.
# with pandas


dict_map = {
    'm1': {'g1': 23, 'g2': 24, 'g3': 15},
    'm2': {'g1': 11, 'g2': 12, 'g3': 15}
}

main_group = ['m1', 'm2']
sub_group = ['g1', 'g2', 'g3']

# Step 1: Convert dict_map to NumPy array
data = np.array([[dict_map[main][sub] for sub in sub_group] for main in main_group], dtype=float)

# Step 2: Create grade array based on sub_group
grade_map = {'g1': 2, 'g2': 3, 'g3': 4}
grades = np.array([grade_map[sub] for sub in sub_group])[None, :]  # shape (1, 3), for broadcasting

# Step 3: Apply transformation and multiply by grade
transformed = ((data + 0.1) ** 1.5) * grades

# Step 4: Compute dense ranks with pandas
flat_transformed = transformed.flatten()
ranks = pd.Series(flat_transformed).rank(method='dense').astype(int).values
rank_array = ranks.reshape(transformed.shape)

# Step 5: Rebuild dict_map with ranks
dict_map_ranked = {
    main: {
        sub: rank_array[i, j]
        for j, sub in enumerate(sub_group)
    }
    for i, main in enumerate(main_group)
}

print(dict_map_ranked)