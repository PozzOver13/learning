import numpy as np

# 1. Create a vector
a = np.array([1, 2, 3], dtype=int)
print(a)
b = np.arange(1, 5, 1, dtype=int)
print(b)
c = np.linspace(1, 5, 5, dtype=int)
print(c)
d = np.zeros(5, dtype=int)
print(d)
e = np.ones(5, dtype=int)
print(e)
f = np.random.randint(low=1, high=10, size=5, dtype=int)
print(f)

# 2. Create a matrix
g = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
print(g)
h = np.arange(1, 7, 1, dtype=int).reshape(2, 3)
print(h)

# 3. Create a pandas DataFrame from a matrix
import pandas as pd
df = pd.DataFrame(g, columns=['a', 'b', 'c'])
print(g)
print(df)

df_stacked_vert = pd.DataFrame(np.vstack((g, h)), columns=['a', 'b', 'c'])
print(df_stacked_vert)

df_stacked_hor = pd.DataFrame(np.hstack((g, h)), columns=['a', 'b', 'c', 'd', 'e', 'f'])
print(df_stacked_hor)

# 4. Extract size from a matrix
print(g.ndim)
print(g.shape)
print(g.size)
