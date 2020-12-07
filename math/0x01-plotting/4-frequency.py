#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
# The bars should be outlined in black
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
# The x-axis should have bins every 10 units
plt.xticks(range(0, 101, 10))
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.show()
