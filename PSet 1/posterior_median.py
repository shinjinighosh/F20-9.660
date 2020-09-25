import matplotlib.pyplot as plt

gamma = 2.43
factor = 2**(1 / gamma)

t_obs = [i for i in range(0, 500, 10)]
t_star = [factor * i for i in t_obs]

plt.figure(1)
plt.plot(t_obs, t_star, marker='o', markerfacecolor='blue', markersize=1,
         color='skyblue', linewidth=4, label="t*")
plt.title("Posterior median as a function of t_obs")
plt.legend()
plt.savefig("posterior_median.png")
plt.show()
