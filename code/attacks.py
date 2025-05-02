import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchattacks
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

from architectures import get_architecture
from datasets import get_dataset, normalize_images, denormalize_images

DEVICE = torch.device("cuda")

# This line modifies the default SSL context used for HTTPS requests to ensure compatibility with servers.
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

ATTACKS = [
    'fgsm',
    'bim',
    'basiciterative',
    'pgd',
    'deepfool',
    'cw',
    'ead',
    'square',
    'autoattack',
    'onepixel',
    'spatialtransformation',
    'jsma',
    'saltandpepper',
    'df',
    'apgd',
    'multiattack',
    'tifgsm'
]

def select_attack(model, attack_type, **kwargs):
    """
    Automatically select and apply an attack from torchattacks based on input parameters.

    Args:
        model: PyTorch model to attack.
        attack_type: A string indicating the type of attack or context (e.g., 'fgsm', 'pgd').
        **kwargs: Additional parameters that influence the choice of attack.

    Returns:
        attack: The selected attack from torchattacks.
    """
    
    # Lowercase the attack_type for standardization
    attack_type = attack_type.lower()
    
    # Initialize the attack variable
    attack = None

    # Select the attack based on attack_type
    if attack_type == 'fgsm':
        epsilon = kwargs.get('epsilon', 0.5)
        attack = torchattacks.FGSM(model, eps=epsilon)
    
    elif attack_type == 'bim' or attack_type == 'basiciterative':
        epsilon = kwargs.get('epsilon', 0.3)
        alpha = kwargs.get('alpha', 0.01)
        steps = kwargs.get('steps', 40)
        attack = torchattacks.BIM(model, eps=epsilon, alpha=alpha, steps=steps)
    
    elif attack_type == 'pgd':
        epsilon = kwargs.get('epsilon', 0.3)
        alpha = kwargs.get('alpha', 2/255)
        steps = kwargs.get('steps', 40)
        attack = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=steps)
    
    elif attack_type == 'deepfool':
        steps = kwargs.get('steps', 50)
        overshoot = kwargs.get('overshoot', 0.02)
        attack = torchattacks.DeepFool(model, steps=steps, overshoot=overshoot)
    
    elif attack_type == 'cw':
        c = kwargs.get('c', 1e-4)
        steps = kwargs.get('steps', 1000)
        lr = kwargs.get('lr', 0.01)
        attack = torchattacks.CW(model, c=c, steps=steps, lr=lr)
    
    elif attack_type == 'ead':
        eps = kwargs.get('eps', 0.3)
        beta = kwargs.get('beta', 0.01)
        steps = kwargs.get('steps', 100)
        attack = torchattacks.EAD(model, eps=eps, beta=beta, steps=steps)
    
    elif attack_type == 'square':
        eps = kwargs.get('eps', 0.3)
        n_queries = kwargs.get('n_queries', 5000)
        attack = torchattacks.Square(model, eps=eps, n_queries=n_queries)
    
    elif attack_type == 'autoattack':
        eps = kwargs.get('eps', 0.3)
        n_classes = kwargs.get('n_classes', 10)
        version = kwargs.get('version', 'standard')
        attack = torchattacks.AutoAttack(model, eps=eps, n_classes=n_classes, version=version)

    elif attack_type == 'onepixel':
        pixels = kwargs.get('pixels', 5)
        steps = kwargs.get('steps', 75)
        popsize = kwargs.get('popsize', 400)
        inf_batch = kwargs.get('inf_batch', 128)
        attack = torchattacks.OnePixel(model, pixels=pixels, steps=steps, popsize=popsize, inf_batch=inf_batch)

    elif attack_type == 'spatialtransformation':
        max_translation = kwargs.get('max_translation', 3)
        num_translations = kwargs.get('num_translations', 20)
        max_rotation = kwargs.get('max_rotation', 30)
        num_rotations = kwargs.get('num_rotations', 30)
        attack = torchattacks.SpatialTransformation(model, max_translation=max_translation, num_translations=num_translations, max_rotation=max_rotation, num_rotations=num_rotations)
    
    elif attack_type == 'jsma':
        theta = kwargs.get('theta', 1.0)
        gamma = kwargs.get('gamma', 0.1)
        attack = torchattacks.JSMA(model, theta=theta, gamma=gamma)
    
    elif attack_type == 'saltandpepper':
        amount = kwargs.get('amount', 0.1)
        attack = torchattacks.SaltAndPepper(model, amount=amount)

    elif attack_type == 'df':
        steps = kwargs.get('steps', 50)
        overshoot = kwargs.get('overshoot', 0.02)
        attack = torchattacks.DeepFool(model, steps=steps, overshoot=overshoot)

    elif attack_type == 'apgd':
        eps = kwargs.get('eps', 0.3)
        n_restarts = kwargs.get('n_restarts', 5)
        n_iter = kwargs.get('n_iter', 100)
        loss = kwargs.get('loss', 'ce')
        attack = torchattacks.APGD(model, eps=eps, n_restarts=n_restarts, n_iter=n_iter, loss=loss)

    elif attack_type == 'multiattack':
        attacks = kwargs.get('attacks', [torchattacks.Square(model), torchattacks.FGSM(model)])
        attack = torchattacks.MultiAttack(attacks)

    elif attack_type == 'tifgsm':
        epsilon = kwargs.get('epsilon', 0.3)
        alpha = kwargs.get('alpha', 0.01)
        steps = kwargs.get('steps', 40)
        decay = kwargs.get('decay', 1.0)
        attack = torchattacks.TIFGSM(model, eps=epsilon, alpha=alpha, steps=steps, decay=decay)
    
    else:

        print(f"Unrecognized attack type '{attack_type}'.")
    
    return attack



def get_adversarial_dataset(model_name: str, dataset_name: str, dataset_split: str, attack_type: str, proportion: float):
    """Return the dataset as a PyTorch Dataset object"""
    if dataset_name == "imagenet":
        return _generate_attacks_ImageNet(model_name, dataset_name, dataset_split, attack_type, proportion)
    elif dataset_name == "cifar10":
        return _generate_attacks_CIFAR10(model_name, dataset_name, dataset_split, attack_type, proportion)
    

class AdversarialDatasetCIFAR10(Dataset):

    def __init__(self, dataset_dict, transform=None):
        """
        Initializes an AdversarialDataset object by generating adversarial examples.

        Args:
            dataset_dict (dict): Dictionary containing original and adversarial images and labels.
            proportion (float): The proportion of adversarial examples in the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = dataset_dict["images"]
        self.labels = dataset_dict["labels"]
        self.transform = transform

    def _generate_adversarial_dataset(self, dataset_dict, proportion = 0.5):
         
        """
         Generates a dataset that contains adversarial examples according to a specific proportion.
     
         Args:
             dataset_dict (dict): The original dataset.
             proportion (float): The proportion of adversarial examples in the dataset.
 
         Returns:
             Tuple of selected images and their corresponding labels.
        """

        # Extract data from the input dictionary
        images = dataset_dict['images']
        adversarial_images = dataset_dict['adversarial_images']
        labels = dataset_dict['labels']

        # Initialize lists for the new dataset
        selected_images = []
        selected_labels = []

        # Iterate over all images and decide whether to include the regular or adversarial version
        for i in range(len(images)):
            if np.random.rand() < proportion:
                # Select adversarial image
                selected_images.append(adversarial_images[i])
            else:
                # Select original image
                selected_images.append(images[i])

            # Append the label
            selected_labels.append(labels[i])

        # Convert the list of images to a numpy array (similar to CIFAR-10 format)
        selected_images = np.array(selected_images)

        return selected_images, selected_labels

    def __len__(self):
        # Return the size of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve a sample and its corresponding label
        sample = self.data[idx]
        label = self.labels[idx]

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)

        return sample, label

# Define the function to generate adversarial attacks
def _generate_attacks_CIFAR10(model_name: str, dataset_name: str, dataset_split: str, attack_type: str, proportion: float):
    """
    Generates adversarial examples for a specified model and dataset.

    Args:
        model_name (str): The name of the model architecture.
        dataset_name (str): The name of the dataset.
        dataset_split (str): The split of the dataset.

    """

    file_path = 'datasets/' + dataset_name + '_' + model_name + '_' + attack_type + '_' + dataset_split + '_' + str(proportion) + '.pt'

    # check if dataset is already there
    if os.path.exists(file_path):

        # return the dataset
        print(f"The file {file_path} exists.")
        dataset_dict = torch.load(file_path)

    else:

        # make directory if it does not exist
        if not os.path.exists('datasets'): os.mkdir('datasets')

        # Load the specified model pre-trained on the specified dataset
        model = get_architecture(model_name, dataset_name)
        model = model.to(DEVICE)

        # Get the dataset
        dataset = get_dataset(dataset_name, dataset_split)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

        # Apply attack
        model.eval()
    
        # You can choose the attack type here
        atk = select_attack(model, attack_type)

        # Generate adversarial dataset
        print("Generating Adversarial Images")

        # Generate adversarial examples and perturbations
        images = []
        adversarial_images = []
        perturbations = []
        labels = []

        for imgs, lbls in loader:

            # denormalize images
            imgs = denormalize_images(dataset_name,imgs)

            # get adversarial attack
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)
            adv_images = atk(imgs, lbls)

            # normalize images
            imgs = normalize_images(dataset_name,imgs)
            adv_images = normalize_images(dataset_name,adv_images)

            prt = adv_images - imgs

            # Append to lists
            images.append(imgs.cpu())
            adversarial_images.append(adv_images.cpu())
            perturbations.append(prt.cpu())
            labels.append(lbls.cpu())

        # Convert lists to tensors
        images = torch.cat(images)
        adversarial_images = torch.cat(adversarial_images)
        perturbations = torch.cat(perturbations)
        labels = torch.cat(labels)

        # return the dataset as dictionary
        dataset_dict = {
            'images': images,
            'adversarial_images': adversarial_images,
            'perturbations': perturbations,
            'labels': labels
        }
        print("Saving to file")
        torch.save(dataset_dict, file_path)

    return dataset_dict


# Define the function to generate adversarial attacks
def _generate_attacks_ImageNet(model_name: str, dataset_name: str, dataset_split: str, attack_type: str, proportion: float):
    """
    Generates adversarial examples for a specified model and dataset.

    Args:
        model_name (str): The name of the model architecture.
        dataset_name (str): The name of the dataset.
        dataset_split (str): The split of the dataset.

    """

    # Get the root path of the dataset
    dataset = get_dataset(dataset_name, dataset_split)
    file_path = 'datasets/' + dataset_name + '_' + model_name + '_' + attack_type + '_' + dataset_split + '_' + str(proportion)

    # check if dataset is already there
    if os.path.exists(file_path):

        # return the dataset
        print(f"The folder {file_path} exists.")

    else:

        # make directory if it does not exist
        if not os.path.exists(file_path): os.mkdir(file_path)

        # Load the specified model pre-trained on the specified dataset
        model = get_architecture(model_name, dataset_name)
        model = model.to(DEVICE)
        model.eval()

        # get classes of the dataset
        class_names = dataset.classes
    
        # You can choose the attack type here
        atk = select_attack(model, attack_type)
        print("Generating Adversarial Images")

        for idx, (image, target) in enumerate(dataset):

            # get image name
            image_path = dataset.imgs[idx][0]
            filename_with_extension = os.path.basename(image_path)

            # get image folder
            subfolder_name = file_path + '/' + class_names[target] 
            if not os.path.exists(subfolder_name): os.mkdir(subfolder_name)
            path = subfolder_name + '/' + filename_with_extension

            # apply adversarial perturbation
            if torch.rand(1).item() < proportion: 
                image = image.unsqueeze(0).to(DEVICE)
                target = torch.tensor([target]).to(DEVICE)
                image = atk(image, torch.tensor([target]))
                image = image.squeeze(0)

            # save image to file
            image = transforms.ToPILImage()(image.cpu()).convert("RGB")
            image.save(path)

    dataset_dict = datasets.ImageFolder(file_path)
    return dataset_dict

    

class PerturbationsDataset(Dataset):

    def __init__(self, dataset_dict, transform=None):
        """
        Initializes an AdversarialDataset object by generating adversarial examples.

        Args:
            dataset_dict (dict): Dictionary containing original and adversarial images and labels.
            proportion (float): The proportion of adversarial examples in the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """        
        
        # added variables
        self.transform = transform
        self.perturbations = self._load_perturbations(dataset_dict)
        self.covariance_matrix = self._compute_covariance_matrix()
        self.distribution = self._create_gaussian_distribution()

        self.data = self._sample_perturbations()

    def __len__(self):
        # Return the size of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve a sample and its corresponding label
        sample = self.data[idx]

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    # added functions
    def _load_perturbations(self, dataset_dict):
        """
        Load perturbations from the .pt file.

        Returns:
        torch.Tensor: The perturbations tensor.
        """
        perturbations = dataset_dict['perturbations']
        return perturbations

    def _compute_covariance_matrix(self):
        """
        Compute the covariance matrix of the flattened perturbations.

        Returns:
        torch.Tensor: The covariance matrix.
        """

        perturbations_flat = self.perturbations.view(self.perturbations.size(0), -1)
        perturbations_np = perturbations_flat.numpy()
        covariance_matrix = np.cov(perturbations_np, rowvar=False)

        # Ensure the covariance matrix is positive definite
        # by adding a small value to the diagonal
        min_eigenvalue = np.min(np.linalg.eigvals(covariance_matrix))
        if min_eigenvalue < 0:
            covariance_matrix += np.eye(covariance_matrix.shape[0]) * (1e-6 - np.real(min_eigenvalue))

        return torch.tensor(covariance_matrix, dtype=torch.float32)
    
    def _create_gaussian_distribution(self):
        """
        Return the Cholesky decomposition of the covariance matrix.

        Returns:
        MultivariateNormal: The multivariate normal distribution.
        """

        mean = torch.zeros(self.covariance_matrix.shape[0], dtype=torch.float32)
        distribution = MultivariateNormal(loc=mean.to(DEVICE), covariance_matrix=self.covariance_matrix.to(DEVICE))
        return distribution
    
    def _sample_perturbations(self):
        """
        Sample new perturbations from the Gaussian distribution.

        Returns:
        torch.Tensor: The sampled images, reshaped to original image dimensions.
        """

        # Generate batch of an original image
        num_samples = len(self.perturbations)
        perturbations = self.distribution.sample((num_samples,))
        perturbations = perturbations.view(num_samples, *self.perturbations.shape[1:])
        perturbations = perturbations.to(DEVICE)

        return perturbations