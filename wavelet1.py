import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg



# %% start point
# xpoints = np.array([1, 10])
ypoints_1 = np.array([1, 3, 6, 2, 6, 83, 66])
ypoints_2 = np.array([3, 3, 6, 7, 3, 7, 8, 9, 4, 33])
colors = np.array(range(1, 700, 100))

# %% end poitn

# plt.plot(ypoints, 'o-.', ms = '16', mec = 'k', mfc = 'hotpink')
# plt.plot(ypoints, ':', ms = '16', mec = 'k', mfc = 'hotpink')
plt.plot(ypoints_1, linestyle="--", color="r")
plt.plot(ypoints_2, linestyle="--", color="k")

plt.scatter(range(len(ypoints_1)), ypoints_1, c=colors, cmap="viridis")

plt.title("MY BEST GRAPH", loc="left")
plt.xlabel("death rate")

plt.grid(lw="0.5", color="c")
plt.colorbar()
plt.show()

np.random.normal()

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(3, 4)
print(newarr)
