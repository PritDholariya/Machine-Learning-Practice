import numpy as np

# dimension
d = 100

# e1
e1 = np.zeros(d)
e1[0] = 1

# random vector x
x = np.random.randn(d)

# choose i > 1
i = np.random.randint(1, d)

# random r ~ U(0,1)
r = np.random.uniform(0, 1)

# modified vector
x_new = x.copy()
x_new[i] += r

dot_original = np.dot(e1, x)
dot_modified = np.dot(e1, x_new)

print("i =", i)
print("r =", r)

print("Original dot product =", dot_original)
print("Modified dot product =", dot_modified)

print("Difference =", abs(dot_original - dot_modified))