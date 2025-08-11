# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 21:59:22 2025

@author: cindyx
"""
import pickle  
import json
import pandas as pd
from yake import KeywordExtractor      
from collections import Counter
from summa import keywords
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
import string
import re
import os
from nltk.stem.porter import PorterStemmer
# from nltk.tokenize import word_tokenize
stemmer = PorterStemmer()
import numpy as np
from PyPDF2 import PdfReader
stemmer = PorterStemmer()
from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt


# Import the dataset
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'utf-16', 'cp1252']
for enc in encodings:
    try:
        df = pd.read_csv(r'file path', encoding=enc)
        print(f"Successfully read the file with {enc} encoding.")
        print(df)
        break
    except UnicodeDecodeError:
        print(f"Failed to read the file with {enc} encoding. Trying the next one.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

#------------------------------------------------------------------------------

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
   
#------------------------------------------------------------------------------    
   
def extract_content_before_abstract(document):
    # Find the index of the first occurrence of "abstract"
    abstract_index = document.lower().find("ract")
    
    # If "abstract" is found
    if abstract_index != -1:
        # Extract the content before "abstract"
        content_before_abstract = document[:abstract_index]
        return content_before_abstract.strip()  # Remove leading/trailing whitespace
    else:
        # If "abstract" is not found, return None or handle it according to your needs
        return None    

#------------------------------------------------------------------------------

def extract_title(text):
    # Splitting the text by newline character
    lines = text.strip().split('\n')
    
    # Extracting the first and second lines and removing trailing whitespace
    title = ' '.join([lines[0].strip(), lines[1].strip()])
    
    return title

#------------------------------------------------------------------------------

def add_underscores(text):
    # Split the text into words
    words = text.split()
    
    # Join the words with underscores
    result = "_".join(words)
    
    return result    
#------------------------------------------------------------------------------
def convert_to_lower(text):
    

    return text.lower()

#------------------------------------------------------------------------------

def remove_numbers(text):

    text = re.sub(r'\d+' , '', text)
    
    return text

#------------------------------------------------------------------------------

def remove_http(text):

    text = re.sub("https?://t.co/[A-Za-z0-9]*", ' ', text)
    
    return text

#------------------------------------------------------------------------------

def remove_short_words(text):

    text = re.sub(r'bw{1,2}b', '', text)
    
    return text

#------------------------------------------------------------------------------

def remove_punctuation(text):
    punctuations = '''!()[]{};«№»:'",`./?@=#$-(%^)+&[*_]~'''
    no_punctuation = ""
    for char in text:
        if char not in punctuations:
            no_punctuation = no_punctuation + char
    return no_punctuation

#------------------------------------------------------------------------------

def remove_white_space(text):

    text = text.strip()
    
    return text

#------------------------------------------------------------------------------

def toknizing(text):
    stop_words = set(stopwords.words('english'))
    # Additional stop words and verbs to be removed
    additional_stop_words = ['result','task','good','need','so''use','two','case','style','brief','lock','pause','score','organizing'
                              ,'selection','successful','manage','important','visualize','manager','understand','home','type','latent','chart',
                              'input','effort','research','paper','data','printer','growth','source','food','analysis','presentation','new','different']
    
    
    stop_words.update(additional_stop_words)

    tokens = word_tokenize(text)
    # Remove stop words and verbs from tokens
    result = [i for i in tokens if i.lower() not in stop_words]
    
    return result
    
    
def process_text(text):

    text = remove_numbers(text)
    
    text = remove_http(text)
    
    text = convert_to_lower(text)
    
    text = remove_white_space(text)
    
    text = remove_short_words(text)
    
    tokens = toknizing(text)
    
    pos_map = {'J': 'a', 'N': 'n'}
    
    pos_tags_list = pos_tag(tokens)
    
    
    lemmatiser = WordNetLemmatizer()
    
    # Filter out tokens that are not adjectives or nouns
    tokens = [lemmatiser.lemmatize(w.lower(), pos=pos_map.get(p[0], 'n')) for w, p in pos_tags_list if p[0] in pos_map]

    return ' '.join(tokens)


numOfKeywords = 10
kw_extractor = KeywordExtractor(top = numOfKeywords)

# Apply the preprocess_text function to the 'abstract' column
df['Abstract'] = df['Abstract'].apply(process_text)

# Use RAKE to extract the keywords
Yake_list = []
for document_text in df['Abstract']:
    document_keywords = kw_extractor.extract_keywords(document_text)
    top_keywords = [keyword for keyword, score in document_keywords]
    Yake_list.extend(top_keywords)  # Append individual keywords to the list
    

# Use TextRank to extract the keywords
TopN = 10
keyword_list = []
for document_text in df['Abstract']:
    document_keywords = keywords.keywords(document_text, words=TopN).split('\n')
    keyword_list.extend(document_keywords)  # Append individual keywords to the list
    
    

#Use TFIDF to extract the keywords

vectorizer = TfidfVectorizer(use_idf=True, max_df=0.5,min_df=1, ngram_range=(1,3))
vectors = vectorizer.fit_transform(df['Abstract'])

dict_of_tokens={i[1]:i[0] for i in vectorizer.vocabulary_.items()}
tfidf_vectors = []  # all deoc vectors by tfidf
for row in vectors:
  tfidf_vectors.append({dict_of_tokens[column]:value for (column,value) in zip(row.indices,row.data)})
  
doc_sorted_tfidfs =[]  # list of doc features each with tfidf weight
#sort each dict of a document
for dn in tfidf_vectors:
  newD = sorted(dn.items(), key=lambda x: x[1], reverse=True)
  newD = dict(newD)
  doc_sorted_tfidfs.append(newD)

tfidf_kw = [] # get the keyphrases as a list of names without tfidf values
for doc_tfidf in doc_sorted_tfidfs:
    ll = list(doc_tfidf.keys())
    tfidf_kw.append(ll)
    
TopN = 10
keyword_list_tfidf = []
for document_keywords in tfidf_kw:
    top_keywords = document_keywords[:TopN]
    # unique_keywords = list(set(top_keywords))
    keyword_list_tfidf.extend(top_keywords)  # Append individual keywords to the list

total_keyword = keyword_list + keyword_list_tfidf + Yake_list


total_keyword_underscores = ["_".join(keyword.split()) for keyword in total_keyword]

# Manually filter out many of the common words

words_to_remove = ['result','task','good','need','so','use','two','case','style','brief','lock','pause','score','organizing'
                          ,'selection','successful','manage','important','visualize','manager','understand','home','type','latent','chart',
                          'input','effort','research','paper','data','printer','growth','source','food','analysis','study','product_development','new','presentation',
                          'development','collaborative','work','go','member','statement','smart','lead','wind','employee','face','agency','gap','impactful','interaction','tolerancing','scan','appropriateness','clarification',
                          'coordination','big','time','low','high','opportunity','life','customer','size','relative','similar','location','system','view','open','us', 'direct', 'goal', 'toys','maker','layout','review',
                          'simple','actor','system','large','number','agreement','field','perform','existence','time','pressure','beam','skill','problem','value','conversation','elevation','assistance', 'practical','perform','firm','organization','actor','framing',
                          'model','product','process','behavior','different','virtual','collage','design_education','formation','collective','performing','resource','communication','senior','learning','people','company','individual',
                          'feedback','user','insight','experience','role','context','generate','positive','exercise','least','usefulness','effect','demonstrate','experiment','impact','metric','addition','level','example',
                          'child','zone','causal','signature','rate','diameter','mode','factor','great','contact','iteration','balance','map','core','solving','flow','approach','method','edge','agent','organize','attribute',
                          'change','lack','challenge','activity','digital','significant_positive_correlation','products','alarm','cluster','devices','failure','energy','feature','coefficient','disadvantage','element',
                          'print','short','today','future','human','interact','making','necessary','professional','seek','social','reuse','users','center','game','interval','search','theory','female','group','mm','survery',
                          'depth','blade','situation','climate','automation','mass','novice','phase','impact_location','preliminary','function','requirement','rudder','region','wind_turbine','controller','safety','paper_presents','Design','in','methods','total',
                          'positive_correlation']

total_keyword_underscores = [word for word in total_keyword_underscores if word not in words_to_remove]

# replace tokenized text with keyword pairs in the keyword list
def replace_keywords(tokenized_text, keyword_list):
    for i in range(len(tokenized_text)):
        sentence = tokenized_text[i]
        j = 0
        while j < len(sentence):
            for keyword in keyword_list:
                words = keyword.lower().split()
                if [word.lower() for word in sentence[j:j+len(words)]] == words:
                    sentence[j:j+len(words)] = [keyword]
                    j += len(words)
                    break
            else:
                j += 1


#Train the word2vec moodel
# # Load the data from the pickle file
# with open('sentences_final.pkl', 'rb') as file:
#     sentences = pickle.load(file)

# replace_keywords(sentences, keyword_list)
# Build word2vec model
# model = Word2Vec(sentences, min_count =3 , vector_size = 300, window = 7, sg = 1)
# model.build_vocab(sentences)  # prepare the model vocabulary
# model.train(sentences, total_examples=model.corpus_count, epochs= model.epochs)  # train word vectors
# model.save("word2vec.modelabstr-more _papers_underscore_2")

#Load the model
model = Word2Vec.load("word2vec.modelabstr-more _papers_underscore_2")

for word in total_keyword_underscores.copy():
    if word not in model.wv.key_to_index:
        total_keyword_underscores.remove(word) 

#Find the three words that are most similar to "product"
w1 = ["product"]

test = model.wv.most_similar(positive = w1, topn=3)
print(test)


unique_words = list(dict.fromkeys(total_keyword_underscores))

# Calculate the word vector for each keyword
vector_list = []
for word in unique_words:
    vector = model.wv[word]
    vector_list.append(vector)
    
# Convert vector list to numpy array
vectors = np.array(vector_list)

# Compute similarity matrix using cosine similarity
similarity_matrix = cosine_similarity(vectors)

print(similarity_matrix)

# Create a graph object
G = nx.Graph()

# Count the occurrence of each keyword
keyword_counts = Counter(keyword_list)


# Add nodes to the graph with their corresponding attributes
num_nodes = similarity_matrix.shape[0]
for i in range(num_nodes):
    G.add_node(i, keyword=unique_words[i])

# Add edges based on similarity matrix
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        similarity = similarity_matrix[i, j]
        if similarity > 0.8:
            G.add_edge(i, j, weight=similarity)

# Determine node sizes based on the occurrence of keywords
node_sizes = [200 * keyword_counts[word] for word in unique_words]

# Determine node positions using spring layout
pos = nx.spring_layout(G)

# Draw the graph with node attributes as labels
plt.figure(figsize=(12, 9))


node_labels = nx.get_node_attributes(G, 'keyword')
# edge_labels = nx.get_edge_attributes(G, 'weight')

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue')
nx.draw_networkx_labels(G, pos, labels=node_labels)
nx.draw_networkx_edges(G, pos)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.axis('off')
plt.show()


# Save the graph in pajek form and open in vosviewer      
nx.write_pajek(G, "keyword_network.net")

# Save the graph in the Pajek format with weights and keywords
with open("keyword_network.net", "w",encoding="utf-8") as f:
    f.write("*Vertices {}\n".format(num_nodes))
    for node in G.nodes:
        keyword = G.nodes[node]['keyword']
        f.write("{} \"{}\"\n".format(node + 1, keyword))

    f.write("*Edges\n")
    for u, v, attr in G.edges(data=True):
        # print(f"({u}, {v}) with attributes {attr}")
        if 'weight' in attr:
            weight = attr['weight']
            # print(f"({u}, {v}) with attributes {weight}")
            f.write("{} {} {:.2f}\n".format(u + 1, v + 1, weight))

# Vosviwer will automatically detect clusters within the graph, save the new graph with the detected clusters and import here

file_path = 'C:/Users/keyword_network.txt'

# Read the .txt file into a DataFrame
df2018 = pd.read_csv(file_path, sep='\t')


list_of_dicts = df2018.to_dict(orient='records')

labels = [item['label'] for item in list_of_dicts]

from collections import defaultdict
# Extract labels and clusters
labels_clusters = [(item['label'], item['cluster']) for item in list_of_dicts]

# Create communities based on cluster assignments
communities = defaultdict(list)
# Iterate through the input list and populate the dictionary
for index, (item, community) in enumerate(labels_clusters):
    communities[community].append(item) 

# Convert the dictionary to a list of lists
clusters = list(communities.values())

def get_key_from_value(value, dictionary):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # Return None instead of a string

def retrieve_keys_for_keywords(keywords, node_labels):
    communities = []
    for keyword in keywords:
        key = get_key_from_value(keyword, node_labels)
        if key is not None:
            communities.append(key)
        
    return communities


communities = [retrieve_keys_for_keywords(cluster, node_labels) for cluster in clusters]


# Flatten the list of lists into a single list
community_indices = set(index for sublist in communities for index in sublist)

# Remove nodes not in the community indices
nodes_to_remove = [node for node in G.nodes if node not in community_indices]
G.remove_nodes_from(nodes_to_remove)

# Filter edges to keep only those where both nodes are in the community indices
edges_to_keep = [(u, v) for u, v in G.edges if u in community_indices and v in community_indices]
G.remove_edges_from([(u, v) for u, v in G.edges if (u, v) not in edges_to_keep])


def set_node_community(G, communities, keywords):
    '''Add community and keywords to node attributes'''
    community_assignments = {}  # Store community assignments
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1
            G.nodes[v]['keywords'] = keywords[v]  # Assign keywords to the node
            # Store node's community assignment
            community_assignments[v] = c + 1
    return community_assignments

def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0

           
def get_color(i, r_off=1, g_off=1, b_off=1):
    '''Assign a color to a vertex.'''
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)

# Set node and edge communities
set_node_community(G, communities, unique_words)
set_edge_community(G)

# Retrieve node attributes
node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]
node_labels = {v: G.nodes[v]['keywords'] for v in G.nodes}  # Assign keywords as labels


# Set community color for edges between members of the same community (internal) and intra-community edges (external)
external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
internal_color = ['black' for e in internal]

comm_pos = nx.spring_layout(G)

plt.rcParams.update({'figure.figsize': (15, 10)})

community_assignments = set_node_community(G, communities, unique_words)


###############################################################################
# Now calculate the Z-Score of each node
import math

# Initialize community variables
n = len(communities)  
communities = [[] for _ in range(n)]

# Iterate over nodes and store them in respective communities
for node, community in community_assignments.items():
    communities[community - 1].append(node)

# Print the communities
for i, community in enumerate(communities):
    print(f"Community {i+1}:", community)


# Function to calculate the value B
def calculate_B(graph, community):
    B = 0
    for node in community:
        Nj = sum([graph[node][n]['weight'] for n in graph.neighbors(node) if n in community])
        B += Nj
    return B

# Calculate B for the given graph and each community
Bs = [calculate_B(G, community) for community in communities]

# Function to calculate the value Q
def calculate_Q(graph, community):
    Q = 0
    for node in community:
        Nj = (sum([graph[node][n]['weight'] for n in graph.neighbors(node) if n in community]))**2
        Q += Nj
    return Q

# Calculate Q for the given graph and each community
Qs = [calculate_Q(G, community) for community in communities]

# Calculate the number of nodes in each community
Ms = [len(community) for community in communities]

# Function to calculate the value Ni
def calculate_Ni(graph, node, community):
    Ni = sum([graph[node][n]['weight'] for n in graph.neighbors(node) if n in community])
    return Ni

# Calculate Ni values for each node in each community
Nis = [{node: calculate_Ni(G, node, community) for node in community} for community in communities]


def calculate_zi(Ni, B, M, Q):
    numerator = Ni - B/M
    denominator = math.sqrt(Q/M - (B/M)**2)
    if denominator != 0:
        zi = numerator / denominator
    else:
        zi = 0  # or any other value you want to assign in case of zero denominator
    return zi


# Calculate zi values for each node  in each community
zis = [{node: calculate_zi(Ni, B, M, Q) for node, Ni in community_Nis.items()} for community_Nis, B, M, Q in zip(Nis, Bs, Ms, Qs)]

# Sort nodes within each community based on zi values
sorted_communities = [sorted(zi_community, key=zi_community.get, reverse=True) for zi_community in zis]

# Set the node with the highest zi value as the label for each community
community_labels18= [node_labels[sorted_community[0]] for sorted_community in sorted_communities]

# Print sorted communities with node labels and zi values
for i, sorted_community in enumerate(sorted_communities):
    print(f"Sorted Community {i+1}:")
    for node  in sorted_community:
        zi_value = zis[i][node]
        label = node_labels[node]
        print(f"Node: {label}")
    print("Community Label:", community_labels18[i])

# Create core node list for each community
core_nodes = [[node for node in sorted_community if zis[i][node] > -5] for i, sorted_community in enumerate(sorted_communities)]

# Print core node lists for each community
for i, core_nodes_community in enumerate(core_nodes):
    print(f"Core Nodes Community {i+1}:", core_nodes_community)

# Normalize Z-Score
all_values = []
for d in zis:
    all_values.extend(d.values())

min_val = min(all_values)
max_val = max(all_values)

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

normalized_data = [
    {key: normalize(value, min_val, max_val) for key, value in d.items()}
    for d in zis
]

normalized_values_list = [list(d.values()) for d in normalized_data]

normalized_z_scores18= normalized_values_list

# Function to retrieve keywords and Word2Vec vectors for core nodes
def retrieve_keywords_and_vectors(core_nodes, node_labels,model):
    core_node_data = []
    for node in core_nodes:
        keyword = node_labels[node]
        try:
            vector = model.wv[keyword]
        except KeyError:
            # Handle KeyError: set vector to zeros if keyword is not present in the model
            vector = np.zeros(model.vector_size)  # or np.zeros(model.vector_size) if using numpy

        core_node_data.append((node, keyword, vector))
    return core_node_data

# Retrieve keywords and vectors for core nodes in each community
core_node_data18 = [retrieve_keywords_and_vectors(core_nodes_community, node_labels,model) for core_nodes_community in core_nodes]



n = len(communities)  
# Save core node lists for each community
with open('18data.pickle', 'wb') as file:
   
    for i in range(n):
        pickle.dump(normalized_z_scores18[i], file)
        pickle.dump(core_node_data18[i], file)
        pickle.dump(community_labels18[i], file)
        
# Repeat the same process for the remaining four years