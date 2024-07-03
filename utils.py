from settings import *

def create_folder():
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    # os.makedirs(os.path.join(store_path, run_name), exist_ok=True)
    # os.makedirs(os.path.join(results_path, run_name), exist_ok=True)
    
def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
    
def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
    
def sample_500(ddpm, ema=True, images_per_class=10):
  if not os.path.exists(CLASS_LEGEND_FILE):
    print("File labels non trovato!")
    exit(1)

  os.makedirs(OUTPUT_PATH, exist_ok=True)

  with open(CLASS_LEGEND_FILE) as json_file:
    legend_labels = json.load(json_file)

  letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  with open(TEST_FILE, 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines, position=0):
        image_name, label = line.strip().split(';')
        print("Classe: ", label)
        label_number = legend_labels[str(int(label, 2))]
        labels = (torch.ones(images_per_class)*label_number).long().to(DEVICE)
        sampled_images = ddpm.sample(n=len(labels), labels=labels, ema=ema)
        for i, image in tqdm(enumerate(sampled_images), position=0):
            image_name_output = image_name + "_" + letters[i]
            file_img = os.path.join(OUTPUT_PATH, f"{image_name_output}.jpg")
            save_images(image, file_img)
    
def get_data():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2)]
    )
    dataset = CustomDataset(DATASET_PATH, transform=transform)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    return loader    
    
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._read_txt_file(os.path.join(root_dir,TRAIN_FILE))

    def _read_txt_file(self, txt_file):
        data = []
        labels = {}
        label_corr = 0
        get_label = False
        if os.exists(CLASS_LEGEND_FILE):
            print("File labels caricato!")
            get_label = True
            with open(CLASS_LEGEND_FILE) as json_file:
                labels = json.load(json_file)
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_name, label = line.strip().split(';')
                image_path = os.path.join(self.root_dir, image_name)
                val = int(label, 2)
                if not get_label:
                  if val not in labels:
                    labels[val] = label_corr
                    label_corr += 1
                else:
                    val = str(val)
                data.append((image_path, labels[val]))
        if not get_label:
            with open(CLASS_LEGEND_FILE, 'w') as f:
                json.dump(labels, f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

