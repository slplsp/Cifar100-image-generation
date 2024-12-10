import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm

superclass_mapping = {
    0: [4, 30, 55, 72, 95],  # aquatic mammals
    1: [1, 32, 67, 73, 91],  # fish
    2: [54, 62, 70, 82, 92],  # flowers
    3: [9, 10, 16, 28, 61],  # food containers
    4: [0, 51, 53, 57, 83],  # fruit and vegetables
    5: [22, 39, 40, 86, 87],  # household electrical devices
    6: [5, 20, 25, 84, 94],  # household furniture
    7: [6, 7, 14, 18, 24],  # insects
    8: [3, 42, 43, 88, 97],  # large carnivores
    9: [12, 17, 37, 68, 76],  # large man-made outdoor things
    10: [23, 33, 49, 60, 71],  # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],  # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],  # medium-sized mammals
    13: [26, 45, 77, 79, 99],  # non-insect invertebrates
    14: [2, 11, 35, 46, 98],  # people
    15: [27, 29, 44, 78, 93],  # reptiles
    16: [36, 50, 65, 74, 80],  # small mammals
    17: [47, 52, 56, 59, 96],  # trees
    18: [8, 13, 48, 58, 90],  # vehicles 1
    19: [41, 69, 81, 85, 89]   # vehicles 2
}

def get_class_to_superclass():
    class_to_superclass = [None] * 100
    for super_idx, class_indices in superclass_mapping.items():
        for class_idx in class_indices:
            class_to_superclass[class_idx] = super_idx
    return class_to_superclass

def load_inception_net():
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model = WrapInception(inception_model.eval()).cuda()
    return inception_model

class WrapInception(nn.Module):
    def __init__(self, net):
        super(WrapInception, self).__init__()
        self.net = net
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                      requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                     requires_grad=False)
    def forward(self, x):
        # Normalize x
        x = (x + 1.) / 2.0
        x = (x - self.mean) / self.std
        # Upsample if necessary
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        # Forward pass through Inception model layers
        x = self.net.Conv2d_1a_3x3(x)
        x = self.net.Conv2d_2a_3x3(x)
        x = self.net.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.net.Conv2d_3b_1x1(x)
        x = self.net.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.net.Mixed_5b(x)
        x = self.net.Mixed_5c(x)
        x = self.net.Mixed_5d(x)
        x = self.net.Mixed_6a(x)
        x = self.net.Mixed_6b(x)
        x = self.net.Mixed_6c(x)
        x = self.net.Mixed_6d(x)
        x = self.net.Mixed_6e(x)
        x = self.net.Mixed_7a(x)
        x = self.net.Mixed_7b(x)
        x = self.net.Mixed_7c(x)
        pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
        return pool, logits

def get_net_output(data_loader, net, device):
    pool, logits, labels = [], [], []

    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        with torch.no_grad():
            pool_val, logits_val = net(x)
            pool += [pool_val.cpu().numpy()]
            logits += [F.softmax(logits_val, 1).cpu().numpy()]
            labels += [y.cpu().numpy()]
    pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
    return pool, logits, labels

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def calculate_inception_score(pred, num_splits=10):
    scores = []
    for index in range(num_splits):
        part = pred[(index * pred.shape[0] // num_splits):((index + 1) * pred.shape[0] // num_splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def calculate_intra_fid(pool, logits, labels, g_pool, g_logits, g_labels, change_superclass=True):
    intra_fids = []
    class_to_superclass = get_class_to_superclass()
    super_labels = np.array([class_to_superclass[i] for i in labels])
    
    if change_superclass:
        g_super_labels = np.array([class_to_superclass[i] for i in g_labels])
    else:
        g_super_labels = np.array(g_labels)
    
    for super_idx in range(20):
        mask = (super_labels == super_idx)
        g_mask = (g_super_labels == super_idx)
        
        pool_low = pool[mask]
        g_pool_low = g_pool[g_mask]
        
        # Ensure that we have the expected number of samples
        if len(g_pool_low) != 2500:
            raise ValueError(f"Expected 2500 samples for superclass {super_idx}, but got {len(g_pool_low)}")
        if len(pool_low) == 0 or len(g_pool_low) == 0:
            continue
        
        mu = np.mean(g_pool_low, axis=0)
        sigma = np.cov(g_pool_low, rowvar=False)
        mu_data = np.mean(pool_low, axis=0)
        sigma_data = np.cov(pool_low, rowvar=False)
        
        fid = calculate_fid(mu, sigma, mu_data, sigma_data)
        intra_fids.append(fid)
        
    return np.mean(intra_fids), intra_fids

def evaluate_generator(
    generator_path,
    generator_class,
    dataset_root,
    img_size,
    latent_dim,
    num_classes,
    channels
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the generator model
    generator = generator_class(
        latent_dim=latent_dim,
        num_classes=num_classes,
        channels=channels
    ).to(device)

    # Load the checkpoint
    checkpoint = torch.load(generator_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    # Generate images
    generated_images_list = []
    generated_labels_list = []

    batch_size = 100

    for super_idx in range(20):
        class_indices = superclass_mapping[super_idx]

        total_samples = 2500
        num_generated = 0

        while num_generated < total_samples:

            current_batch_size = min(batch_size, total_samples - num_generated)

            # Sample random noise
            z = torch.randn(current_batch_size, latent_dim, device=device)

            # Sample class labels from the classes in this superclass
            class_labels = np.random.choice(class_indices, size=current_batch_size)
            class_labels_tensor = torch.tensor(class_labels, device=device, dtype=torch.long)

            # Generate images
            with torch.no_grad():
                images = generator(z, class_labels_tensor)
                images = images.to('cpu')

            # Store images and labels
            generated_images_list.append(images)
            generated_labels_list.append(class_labels_tensor.cpu())
            num_generated += current_batch_size

    # After all superclasses, concatenate all
    generated_images = torch.cat(generated_images_list, dim=0)
    generated_labels = torch.cat(generated_labels_list, dim=0)

    # Load CIFAR100-Train dataset
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    cifar100_train = datasets.CIFAR100(root=dataset_root, train=True, transform=transform, download=True)

    train_loader = DataLoader(
        cifar100_train, batch_size=100, shuffle=False, num_workers=2, pin_memory=True
    )

    # Load Inception network
    inception_net = load_inception_net()
    inception_net = inception_net.to(device)
    inception_net.eval()

    # Compute activations for real images
    real_pool, real_logits, real_labels = get_net_output(train_loader, inception_net, device)

    # Create DataLoader for generated images
    generated_dataset = TensorDataset(generated_images, generated_labels)
    generated_loader = DataLoader(
        generated_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True
    )

    # Compute activations for generated images
    generated_pool, generated_logits, generated_labels = get_net_output(
        generated_loader, inception_net, device
    )

    # Compute FID
    mu_real = np.mean(real_pool, axis=0)
    sigma_real = np.cov(real_pool, rowvar=False)

    mu_gen = np.mean(generated_pool, axis=0)
    sigma_gen = np.cov(generated_pool, rowvar=False)

    fid = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)

    # Compute Inception Score
    is_mean, is_std = calculate_inception_score(generated_logits)

    # Compute Intra-FID
    intra_fid_mean, intra_fid_list = calculate_intra_fid(
        real_pool, real_logits, real_labels,
        generated_pool, generated_logits, generated_labels,
        change_superclass=True
    )

    return fid, is_mean, is_std, intra_fid_mean
