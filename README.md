# 🌲 TreeForge — Random Forest Regressor (WIP)

### 🔎 Overview  
**TreeForge** is a from-scratch implementation of a **Random Forest Regressor** built entirely with **NumPy**.  
The project focuses on understanding how ensemble learning models like Random Forests operate at a low level — including tree construction, splitting logic, and recursive learning.

At its current stage, the code implements the **core decision tree logic** that serves as the foundation for the forest.  
Further updates will introduce randomization, bootstrapping, and multi-tree aggregation.

---

### ⚙️ Features  
- 🧩 `TNode` class representing tree nodes  
- 🌿 Recursive `build_tree()` function  
- 🎯 Optimal split selection using Mean Squared Error (MSE)  
- ⛔ Stopping conditions:
  - Minimum samples per split or leaf  
  - Maximum depth limit  
- 🧠 Clear, educational structure for understanding regression trees  

---

### 🧠 How It Works  
1. **Tree Building**  
   The algorithm searches across all features and thresholds to find the split that minimizes MSE.  
2. **Node Creation**  
   Each node stores:
   - `index`: feature index used for the split  
   - `value`: threshold value  
   - `prediction`: mean of the target values  
3. **Recursion**  
   The tree recursively expands until the stopping criteria are met.  
4. **Foundation for Forests**  
   These trees will later be combined using:
   - Bootstrapped data samples  
   - Random feature selection  
   - Averaged predictions across all trees  

---

### 🧩 Example Usage  

```python
import numpy as np
from forest import build_tree  # replace 'forest.py' with your file name

# Example data
X = np.array([[2.3], [1.9], [3.2], [2.7], [3.8]])
y = np.array([1.4, 1.3, 2.1, 1.8, 2.5])

# Build a regression tree
tree = build_tree(X, y, min_samples_split=2, max_depth=3)
