## Decision Tree

- [Gini index and Entropy](https://github.com/yukikitayama/machine-learning/blob/master/decision-tree/gini_index_entropy.ipynb)

### Pure

- Used in classification with decision tree
- A node is pure when it is predominant with a single class
- Gini index and Entropy both gives us the measurement about how data is uniform or diverse in a node.
    - Entropy has a sharper contrast between uniform group and diverse group, but entropy is a bit more computationally
      expensive than Gini. But fundamentally no difference.

### Gini index

- Used in classification with decision tree
- Small when a node contains predominantly single class observations.
- Large when a leaf node contains a variety of classes.
- A node is pure when Gini index is small.
    - The lower the gini, the easier we can assign the right label for the samples in the group.
    - The lower the gini, the more uniform for the samples in each group.
    - Reducing the Gini allows us to reduce the uncertainty of guessing the group.

```python
from collections import Counter
from typing import List

def binary_gini(p: float) -> float:
  return p * (1 - p) + (1 - p) * (1 - (1 - p))

def multi_class_gini(data: List[int]) -> float:
  counter = Counter(data)
  proportions = [c / len(data) for c in counter.values()]
  gini = 0
  for proportion in proportions:
    gini += proportion * (1 - proportion)
  return gini
```

#### Gini Gain
- Reduction of Gini.
- Measure the quality of the split in decision tree.
- The higher Gini gain, the better the split

### Entropy

- Used in classification with decision tree
- Small when a node has predominant single class data, and large when it's diverse.
- A node is pure when the entropy is small.

```python
import math
from collections import Counter
from typing import List

def binary_entropy(p: float) -> float:
    return -1 * (p * math.log(p) + (1 - p) * math.log(1 - p))

def multi_class_entropy(data: List[int]) -> float:
    counter = Counter(data)
    proportions = [c / len(data) for c in counter.values()]
    entropy = 0
    for proportion in proportions:
        entropy += proportion * math.log(proportion)
    entropy *= -1
    return entropy
```

#### Information Gain

- Reduction of entropy.
