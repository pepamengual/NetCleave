import matplotlib.pyplot as plt
import numpy as np



geom_means = np.array([5623, 4403, 3970, 2646, 5477, 1381, 5264])
predictions = np.array([0.9322, 0.8632, 0.8788, 0.7117, 0.8666, 0.5825, 0.8482])

correlation_matrix = np.corrcoef(geom_means, predictions)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2

m, b = np.polyfit(geom_means, predictions, 1)
plt.plot(geom_means, m*geom_means + b, color="black")
plt.scatter(x=geom_means, y=predictions)

plt.text(x=3600, y=0.75, s="$R^2$ = {}".format(round(r_squared, 2)))
plt.xlabel("GeoMean PE-A", size=12)
plt.ylabel("PROcleave score", size=12)
#plt.show()
plt.tight_layout()
plt.savefig("geomeans_procleave.png", dpi=300)
