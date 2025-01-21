import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Read the Loss Values of Each Epoch from File
f = open('Loss_values_18.txt','rb')
x = pickle.load(f)
f.close()

# Convert the Loss Values into Numpy Array
y = torch.tensor(x)
y1 = y.numpy()
n = torch.arange(1,19,1)
n1 = n.numpy()

# Display Loss Plot of Diffusion Model
plt.plot(n1,y1, color='red',linewidth=2.5)
plt.xticks(np.arange(1,19,1))

plt.grid(True,color='black',linewidth=0.3)
plt.xlabel('Number of Epoch', fontsize=16)
plt.ylabel('Mean Squared Error Loss', fontsize=16)
plt.suptitle('Loss Plot of Diffusion Model', fontsize=20)
plt.show()


