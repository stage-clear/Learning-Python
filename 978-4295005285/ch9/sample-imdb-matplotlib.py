import matplotlib.pyplot as plt

episodes = ['S' + e.split('.')[0] if int(e.split('.')[1]) == 1 else '' \
    for e in episodes]

plt.figure()
positions = [a * 2 for in range(len(ratings))]
plt.bar(positions, ratings, aligh='center')
plt.xticks(positions, episodes)
plt.show()
