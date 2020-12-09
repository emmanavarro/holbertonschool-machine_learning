#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Creating the figure object to contain the five graphics
fig = plt.figure()
fig.suptitle('All in One')

# Adding 0-line.py
fig.add_subplot(321)
plt.xlim(0, len(y0)-1)
plt.xticks(range(0, 11, 2))
plt.plot(range(len(y0)), y0, color='red')
plt.yticks(range(0, 1001, 500))

# Adding 1-scatter.py
fig.add_subplot(322)
plt.scatter(x1, y1, s=1, c="magenta")
plt.xlabel("Height (in)", fontsize='x-small')
plt.ylabel("Weight (lbs)", fontsize='x-small')
plt.title("Men's Height vs Weight", fontsize='x-small')

# Adding 2-change_scale.py
fig.add_subplot(323)
plt.plot(x2, y2)
plt.xlabel("Time (years)", fontsize='x-small')
plt.ylabel("Fraction Remaining", fontsize='x-small')
plt.title("Exponential Decay of C-14", fontsize='x-small')
plt.yscale("log")
plt.xlim(0, 28650)

# Adding 3-two.py
fig.add_subplot(324)
plt.xlabel("Time (years)", fontsize='x-small')
plt.ylabel("Fraction Remaining", fontsize='x-small')
plt.title("Exponential Decay of Radioactive Elements", fontsize='x-small')
plt.xlim(0, 20000)
plt.ylim(0, 1)
# x ↦ y1 should be plotted with a dashed red line
plt.plot(x3, y31, 'r--', label='C-14')
# x ↦ y2 should be plotted with a solid green line
plt.plot(x3, y32, 'g-', label='Ra-226')
plt.legend(fontsize='x-small')

# Adding 4-frequency.py
fig.add_subplot(313)
plt.xlabel('Grades', fontsize='x-small')
plt.ylabel('Number of Students', fontsize='x-small')
plt.title('Project A', fontsize='x-small')
# The bars should be outlined in black
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
# The x-axis should have bins every 10 units
plt.xticks(range(0, 101, 10))
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.yticks(range(0, 31, 10))

fig.tight_layout()
plt.show()
