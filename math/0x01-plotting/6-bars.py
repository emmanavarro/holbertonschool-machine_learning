#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

range_bar = range(3)
width = 0.5
apples_bar = plt.bar(range_bar, fruit[0, :], color='red', width=width)
bananas_bar = plt.bar(range_bar, fruit[1, :],
                      bottom=fruit[0, :], color='yellow', width=width)
oranges_bar = plt.bar(range_bar, fruit[2, :],
                      bottom=fruit[0, :] + fruit[1, :],
                      color='#ff8000', width=width)
peaches_bar = plt.bar(range_bar, fruit[3, :],
                      bottom=fruit[0, :] + fruit[1, :] + fruit[2, :],
                      color='#ffe5b4', width=width)

plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.xticks(range_bar, ('Farrah', 'Fred', 'Felicia'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((apples_bar[0], bananas_bar[0], oranges_bar[0], peaches_bar[0]),
           ('Apples', 'Bananas', 'Oranges', 'Peaches'))
plt.show()
