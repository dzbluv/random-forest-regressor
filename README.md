# 🌲 Random Forest Regressor — From Scratch (WIP)

### 🔎 Overview  
This project is a **from-scratch implementation of a Random Forest Regressor** using **NumPy**, built to understand the core mechanisms behind ensemble learning.  

It starts with the foundation — implementing decision trees manually — and will gradually expand into a full random forest model with bootstrapping, random feature selection, and prediction aggregation.

---

### ⚙️ Features  
- Custom `TNode` class representing tree nodes  
- Recursive `build_tree()` function for constructing regression trees  
- `find_best_split()` using Mean Squared Error (MSE)  
- Configurable parameters:
  - Minimum samples per split and leaf  
  - Maximum tree depth  
- Simple and transparent structure for educational purposes  

---

### 🧠 How It Works  
1. **Tree Construction:**  
   For each node, the algorithm searches for the best feature and threshold to minimize MSE.  
2. **Node Representation:**  
   Each node stores:
   - `index`: splitting feature index  
   - `value`: threshold value  
   - `prediction`: mean of the target values in that node  
3. **Recursive Growth:**  
   The tree expands recursively until stopping conditions are met.  
4. **Future Extension:**  
   Multiple trees will later be trained on bootstrapped data and combined to form a Random Forest.

---
...
