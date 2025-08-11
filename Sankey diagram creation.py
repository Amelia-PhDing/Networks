# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 23:44:06 2025

@author: cindyx
"""

import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity 
import itertools 

# Create empty lists to store the data

normalized_z_scores18 = []
core_node_data18 = []
community_labels18 = []


normalized_z_scores19 = []
core_node_data19 = []
community_labels19 = []

normalized_z_scores20 = []
core_node_data20 = []
community_labels20 = []

normalized_z_scores21 = []
core_node_data21 = []
community_labels21 = []

normalized_z_scores22 = []
core_node_data22 = []
community_labels22 = []


# Load the data from the pickle file
with open('18data.pickle', 'rb') as file:
    try:
        while True:
            normalized_z_scores18.append(pickle.load(file))
            core_node_data18.append(pickle.load(file))
            community_labels18.append(pickle.load(file))
    except EOFError:
        pass
    
with open('19data.pickle', 'rb') as file:
    try:
        while True:
            normalized_z_scores19.append(pickle.load(file))
            core_node_data19.append(pickle.load(file))
            community_labels19.append(pickle.load(file))
    except EOFError:
        pass
with open('20data.pickle', 'rb') as file:
    try:
        while True:
            normalized_z_scores20.append(pickle.load(file))
            core_node_data20.append(pickle.load(file))
            community_labels20.append(pickle.load(file))
    except EOFError:
        pass
    
with open('21data.pickle', 'rb') as file:
    try:
        while True:
            normalized_z_scores21.append(pickle.load(file))
            core_node_data21.append(pickle.load(file))
            community_labels21.append(pickle.load(file))
    except EOFError:
        pass
    
with open('22data.pickle', 'rb') as file:
    try:
        while True:
            normalized_z_scores22.append(pickle.load(file))
            core_node_data22.append(pickle.load(file))
            community_labels22.append(pickle.load(file))
    except EOFError:
        pass
    

community_labels2018 = community_labels18

community_labels2019 = community_labels19

community_labels2020 = community_labels20

community_labels2021 = community_labels21

community_labels2022 =  community_labels22


normalized_z_scores18 = [sorted(sublist, reverse=True) for sublist in normalized_z_scores18]
normalized_z_scores19 = [sorted(sublist, reverse=True) for sublist in normalized_z_scores19]
normalized_z_scores20 = [sorted(sublist, reverse=True) for sublist in normalized_z_scores20]
normalized_z_scores21 = [sorted(sublist, reverse=True) for sublist in normalized_z_scores21]
normalized_z_scores22 = [sorted(sublist, reverse=True) for sublist in normalized_z_scores22]


# Years lists containing the year numbers
years = [2018, 2019, 2020, 2021, 2022]

# List containing the names of the normalized z-score lists
z_score_lists = ['normalized_z_scores18', 'normalized_z_scores19', 'normalized_z_scores20', 'normalized_z_scores21', 'normalized_z_scores22']

# List containing the names of the core node data lists
core_node_lists = ['core_node_data18', 'core_node_data19', 'core_node_data20', 'core_node_data21', 'core_node_data22']

# Calculate topic similarity
################################################################################
# Threshold for topic similarity
threshold = 0.81


def calculate_topic_similarity(source_year, source_community, target_year, target_community):
    # Access normalized z-scores and core node data
    normalized_z_scores1 = globals()[z_score_lists[years.index(source_year)]][source_community]
    normalized_z_scores2 = globals()[z_score_lists[years.index(target_year)]][target_community]
    core_node_data1 = globals()[core_node_lists[years.index(source_year)]][source_community]
    core_node_data2 = globals()[core_node_lists[years.index(target_year)]][target_community]

    # Calculate the product of all possible pairs between the two lists
    dot_product_sum = sum(zi * zj for zi, zj in itertools.product(normalized_z_scores1, normalized_z_scores2))

    # Calculate the sum of zwt * zwt+1 * cosine(vector_vwt, vector_vwt+1)
    sum_product = 0
    for i, (z_score_wt, wt) in enumerate(zip(normalized_z_scores1, core_node_data1)):
        vector_vwt = np.array(wt[2])
        for j, (z_score_wt_plus_1, wt_plus_1) in enumerate(zip(normalized_z_scores2, core_node_data2)):
            vector_vwt_plus_1 = np.array(wt_plus_1[2])
            
            # Compute cosine similarity between each pair of vectors
            cosine_similarity_score = cosine_similarity(vector_vwt.reshape(1, -1), vector_vwt_plus_1.reshape(1, -1))[0, 0]
            
            # Calculate the product and accumulate
            product = z_score_wt * z_score_wt_plus_1 * cosine_similarity_score
            sum_product += product

    # Calculate topic similarity
    if dot_product_sum != 0:
        topic_similarity = sum_product / dot_product_sum
    else:
        topic_similarity = 0

    return topic_similarity


# Lists to store the results
topic_similarity_results = []
# Dictionary to store the grouped results
grouped_results = {}

community_labels = {
    2018: community_labels2018,
    2019: community_labels2019,
    2020: community_labels2020,
    2021: community_labels2021,
    2022: community_labels2022
}

# Calculate topic similarity between consecutive years with the same community number and between different community numbers
for i in range(len(years) - 1):  # Iterate up to the second-to-last year
    source_year = years[i]
    target_year = years[i + 1]
    source_communities = community_labels[source_year]
    target_communities = community_labels[target_year]
    
    for source_community in range(len(source_communities)):
            for target_community in range(len(target_communities)):
                
                similarity = calculate_topic_similarity(source_year, source_community, target_year, target_community)
                
                # Print the comparison information, including actual community labels and similarity score
                source_label = community_labels[source_year][source_community]
                target_label = community_labels[target_year][target_community]
                # Append the result to the list (for overall results)
                topic_similarity_results.append((f"{source_label} ({source_year}) -> {target_label} ({target_year})", similarity))

                # Append the result to the grouped results dictionary
                if (source_year, target_year) not in grouped_results:
                    grouped_results[(source_year, target_year)] = []
                grouped_results[(source_year, target_year)].append((source_label, target_label, similarity))

                # Print the result
                # print(f"Topic similarity between {source_year} community '{source_label}' and {target_year} community '{target_label}': {similarity}")
 
# Plot the Sankey diagram
################################################################################
                    
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
# Define a similarity threshold
similarity_threshold = 0.84  # Adjust this value as needed
# Flatten grouped_results to extract source, target, and similarity
source_labels = []
target_labels = []
similarity_values = []


for (source_year, target_year), results in grouped_results.items():
    for source_label, target_label, similarity in results:
        # Include only those similarities greater than the threshold
        if similarity > similarity_threshold:
            source_labels.append(f"{source_label}")
            target_labels.append(f"{target_label}")
            similarity_values.append(similarity)
  
# Check if there are enough connections for the Sankey diagram
if not source_labels or not target_labels or not similarity_values:
    print("No connections above the threshold for the Sankey diagram.")
else:
    # Create a list of unique labels (source and target combined)
    all_labels = list(set(source_labels + target_labels))

    # Create a mapping from labels to index (needed for Sankey)
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    # Convert the source and target labels to indices using the mapping
    sources = [label_to_index[label] for label in source_labels]
    targets = [label_to_index[label] for label in target_labels]
    
    # Generate a distinct color palette
    label_colors = px.colors.qualitative.Plotly  # Default Plotly qualitative colors
    while len(label_colors) < len(all_labels):
        label_colors.extend(label_colors)  # Extend if labels exceed available colors
    label_colors = label_colors[:len(all_labels)]  # Trim to match the number of labels
    
    # Map link colors to the source label colors
    link_colors = [label_colors[source] for source in sources]

    # Create a Sankey diagram using plotly
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,  # Padding between nodes
            thickness=20,  # Thickness of nodes
            line=dict(color="black", width=0.5),
            label=all_labels,  # List of unique labels
            color=label_colors  
        ),
        link=dict(
            source=sources,  # Indices for source nodes
            target=targets,  # Indices for target nodes
            value=similarity_values,  # Similarity scores as link values
            color=link_colors
        )
    ))

    
    
    # Update layout settings
    fig.update_layout(
        title_text="Topic Similarity Sankey Diagram",
        font=dict(size=33, color='black', family="Times New Roman black")
        # font=dict(size=25, color='black', family="Arial Black")
    )


    # Show the diagram
    fig.show()
# Save the Sankey diagram as an HTML file
pyo.plot(fig, filename="topic_similarity_sankey.html", auto_open=True)                  

