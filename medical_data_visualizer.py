import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv('medical_examination.csv')

# 2. Add 'overweight' column based on BMI (Body Mass Index)
# BMI = weight(kg) / (height(m))^2
# A BMI > 25 is considered overweight, so we assign 1 to overweight people, 0 otherwise.
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3. Normalize 'cholesterol' and 'gluc' values:
# If the value is 1 (normal), we convert it to 0 (good)
# If it's more than 1 (above normal or well above), we convert it to 1 (bad)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# 4. Categorical plot function
def draw_cat_plot():
    # Create DataFrame in long format using melt to reshape the data
    df_cat = pd.melt(df,
                     id_vars=['cardio'],  # Keep 'cardio' as identifier
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group the data to get total counts for each category
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # Draw the barplot using seaborn's catplot
    plot = sns.catplot(x='variable', y='total', hue='value', col='cardio',
                       data=df_cat, kind='bar', height=5, aspect=1)

    # Get the figure object for saving or displaying
    fig = plot.fig
    return fig


# 5. Heatmap function
def draw_heat_map():
    # Clean the data by removing inconsistent or extreme values

    # Keep rows where:
    # - diastolic pressure (ap_lo) is less than or equal to systolic pressure (ap_hi)
    # - height is within the 2.5th and 97.5th percentiles
    # - weight is within the 2.5th and 97.5th percentiles
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle of the heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap with annotations
    sns.heatmap(corr,
                mask=mask,
                annot=True,
                fmt=".1f",
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .5})

    return fig
