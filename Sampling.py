import torch
import torch.nn as nn
import Model
from Model import DDPM
import matplotlib.pyplot as plt

number_of_time_steps = 1000

# Select the Device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()

# Initiate object of DDPM
model = DDPM(input_image_channel=1,output_image_channel=1)

# Insert state_dict into the Model
model.load_state_dict(torch.load('/home/idrbt/Desktop/DIFFUSION/DDPM_MODEL_18_Epochs.pth',map_location=device,weights_only=True))

# Keep the Model in Evaluation Mode
model.eval()
model.to(device)

# Values of Beta, Alpha, Alpha_Ber
beta = torch.linspace(1e-4,0.02,number_of_time_steps).to(device)
alpha = (1-beta).to(device)
alpha_ber = torch.cumprod(alpha, dim=0).to(device)

# Generate Time Index t
t = []
for i in range(1,number_of_time_steps,1):
    t.append(i)
t.reverse()

# Generate Input Random Gaussian Noise
x = torch.randn(1,1,32,32).to(device)

# Sampling Loop
with torch.no_grad():
    for i in t:

        if i > 1:
            z = torch.randn(x.shape,device=device)
        else:
            z = torch.zeros(x.shape,device=device)

        p = torch.tensor([i],device=device)

        beta_required = beta[p]
        alpha_required = alpha[p]
        alpha_ber_required = alpha_ber[p]

        a = 1/torch.sqrt(alpha_required)
        b = (1-alpha_required)/torch.sqrt(1-alpha_ber_required)
        sigma = torch.sqrt(beta_required)

        model_output = model(x,p)

        x = a*(x-(b*(model_output))) + sigma*z


# Scale the Generated Tensor into 0 to 1 range
d = torch.max(x)-torch.min(x)
y = (x-torch.min(x))/d

# Convert the Generated Tensor into Numpy Array
y1 = y.to('cpu').detach().numpy()

# Display the Generated Image
plt.imshow(y1[0][0])
plt.show()


