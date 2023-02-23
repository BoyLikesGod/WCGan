from matplotlib import pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6])
r = np.array([445, 499, 256, 457, 202, 76])
nr = np.array([1549, 668, 748, 398, 201, 61])
rnr = r/nr
acc = np.array([0.874, 0.8342, 0.8663, 0.8625, 0.7143, 0.8276])
f1 = np.array([0.8033, 0.7697, 0.8659, 0.8612, 0.7118, 0.8222])
pre = np.array([0.8382, 0.792, 0.8739, 0.8627, 0.7116, 0.8291])
recall = np.array([0.7801, 0.7546, 0.8681, 0.8602, 0.7156, 0.8187])

plt.plot(x, rnr, x, recall/f1)
plt.show()