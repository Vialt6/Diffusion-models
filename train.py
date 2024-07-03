from settings import *
from models import *
from utils import *
from ddpm import *

def training_loop(display=True):
    create_folder()
    dataloader = get_data()
    ddpm = MyDDPM()
    mse = nn.MSELoss()    
    optimizer = optim.AdamW(ddpm.parameters(), lr=LR)    # DA CONTROLLARE ddpm.parameters()
    file_optimizer = os.path.join(MODEL_PATH, OPTIM_FILE)
    if os.exists(file_optimizer):
        optimizer.load_state_dict(torch.load(file_optimizer))
        print("Optimizer caricato!")
    
    #get previous number of epoch
    start_epoch = 0
    if os.exists(NUM_EPOCH_FILE):
        with open(NUM_EPOCH_FILE) as fp:
            start_epoch = int(fp.read()) + 1
            print(f"Caricata epoca precendete = {start_epoch}")

    for epoch in range(start_epoch, N_EPOCHS):
        epoch_loss = 0.0
        print(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            t = ddpm.sample_timesteps(images.shape[0]).to(DEVICE)
            x_t, noise = ddpm(images, t)
            if np.random.random() < 0.1:
                labels = None
            
            predicted_noise = ddpm.backward(x_t, t, labels) # VERIFICARE SE NON t.reshape(n, -1) al posto di t
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ddpm.step_ema()

            pbar.set_postfix(MSE=loss.item())
            epoch_loss += loss.item() * len(images) / len(dataloader.dataset)

        print(f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}")

        ddpm.save_model()
        ddpm.save_ema_model()
        torch.save(optimizer.state_dict(), file_optimizer)
        with open(NUM_EPOCH_FILE, 'w') as fp:
            fp.write(str(epoch))

        if display and epoch % 10 == 0:
            labels = torch.arange(5).long().to(DEVICE)
            sampled_images = ddpm.sample(n=len(labels), labels=labels)
            ema_sampled_images = ddpm.sample(n=len(labels), labels=labels, ema=True)
            plot_images(sampled_images)
            plot_images(ema_sampled_images)
            file_img = os.path.join(RESULTS_PATH, f"{epoch}.jpg")
            file_img_ema = os.path.join(RESULTS_PATH, f"{epoch}_ema.jpg")
            save_images(sampled_images, file_img)
            save_images(ema_sampled_images, file_img_ema)
            