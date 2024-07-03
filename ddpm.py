from settings import *
from models import *

# DDPM class
class MyDDPM(nn.Module):
    def __init__(self):
        super(MyDDPM, self).__init__()
        self.network = UNet().to(DEVICE)
        self.file_network = os.path.join(MODEL_PATH, MODEL_FILE)
        if os.path.exists(self.file_network):
            self.load_model()
        
        self.ema = EMA()
        self.ema_network = copy.deepcopy(self.network).eval().requires_grad_(False)
        self.file_ema_network = os.path.join(MODEL_PATH, EMA_MODEL_FILE)
        if os.path.exists(self.file_ema_network):
            self.load_ema_model()

        self.beta = self.noise_schedule().to(DEVICE)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def noise_schedule(self):
        return torch.linspace(BETA_START, BETA_END, NOISE_STEPS)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=NOISE_STEPS, size=(n,))
    
    def sample(self, n, labels, ema=False, cfg_scale=3):
        print(f"Sampling {n} new images....")
        model = self.ema_network if ema else self.network
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, IMAGE_CHW[0], IMAGE_CHW[1], IMAGE_CHW[2])).to(DEVICE)
            for i in tqdm(reversed(range(1, NOISE_STEPS)), position=0):
                t = (torch.ones(n) * i).long().to(DEVICE)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = self.backward(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                alpha_hat_prec = self.alpha_hat[t-1][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                #x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                val = 1/torch.sqrt(alpha_hat) * x - torch.sqrt((1-alpha_hat)/alpha_hat) * predicted_noise
                x = (torch.sqrt(alpha_hat_prec)*beta / (1 - alpha_hat)) * val.clamp(-1,1) + (1-alpha_hat_prec)*torch.sqrt(alpha)/(1-alpha_hat) * x + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
    def load_model(self):
        self.network.load_state_dict(torch.load(self.file_network))
        print("Modello caricato!")
        
    def load_ema_model(self):
        self.ema_network.load_state_dict(torch.load(self.file_ema_network))
        print("Modello EMA caricato!")
        
    def save_model(self):
        torch.save(self.network.state_dict(), self.file_network)
        print("Modello salvato!")
        
    def save_ema_model(self):
        torch.save(self.ema_network.state_dict(), self.file_ema_network)
        print("Modello EMA salvato!")
            
    def step_ema(self):
        self.ema.step_ema(self.ema_network, self.network)

    def forward(self, x0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * eps, eps

    def backward(self, x, t, c=None):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t, c)