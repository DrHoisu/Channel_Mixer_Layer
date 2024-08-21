import numpy as np


arrange = np.arange(72)
print(arrange.shape)

latitude = arrange.reshape(72,1)
latitude = np.repeat(latitude, repeats=72, axis=1)/72
print(latitude)

longitude = arrange.reshape(1,72)
longitude = np.repeat(longitude, repeats=72, axis=0)/72
print(longitude)
