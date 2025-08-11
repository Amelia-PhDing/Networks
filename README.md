# Topic Evolution Network Analysis with Cosine Similarity and Sankey Visualization

This project analyzes how research topics evolve over time by:
1. Creating keyword networks based on **cosine similarity**.
2. Calculating **normalized z-scores** to determine the most important node within a community in the netowrk.
3. Detecting communities using **VOSviewer**.
4. Visualizing topic evolution over multiple years with a **Sankey diagram**.

The workflow requires both Python scripts and external tools (Pajek and VOSviewer) to complete the analysis.

---

## Features

- **Cosine Similarity-Based Network Creation** – Build weighted topic networks from keyword vectors.
- **External Tool Integration** – Export networks for Pajek and VOSviewer processing.
- **Community Detection** – Use VOSviewer’s algorithms to determine communities.
- **Z-score Calculation** – Calculate normalized Z-Score for each node.
- **Sankey Diagram** – Visualize how communities evolve across multiple years.

---

## Prerequisites

This workflow requires **both Python and external network analysis tools**:

- **Python 3.10+** with:
  - `pandas`
  - `numpy`
  - `networkx`
  - `plotly`
  - `scikit-learn`
  - `Pajek`
- **VOSviewer** ([Download](https://www.vosviewer.com/))

---

## Workflow Overview

The analysis consists of two main Python scripts:

1. **`ZScore calculation.py`**  
   Creates cosine similarity networks, saves in Pajek format, processes community assignments, calculate Z-Score, and saves all these values in a pickle file

2. **`Sankey diagram creation.py`**  
   Reads processed z-score data and generates a Sankey diagram showing topic evolution.

---


