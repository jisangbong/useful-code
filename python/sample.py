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