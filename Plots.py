import matplotlib.pyplot as plt

train_losses, validation_losses = [], []
iterations = []
plt.plot(iterations, train_losses, label="Train", color='b')
plt.plot(iterations, validation_losses, label="validation", color='r')
plt.xlabel("Iteration")
plt.ylabel('Loss')
plt.legend()
plt.show()
