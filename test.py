import helper_module
import numpy as np


anchor_pos = helper_module.getRandomCoordsAroundPoint(5, 0.0, 0.0, 0.0, 5, True)

print(anchor_pos)

# Check that all vectors have a length of 5 (the radius)
lengths = np.linalg.norm(anchor_pos - np.array([0.0, 0.0, 0.0]), axis=1)
print(lengths)