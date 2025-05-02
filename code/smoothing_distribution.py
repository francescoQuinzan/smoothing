import torch

DEVICE = torch.device("cuda")

class GaussianImageSampler:

    def __init__(self, data, sigma):
        """
        Initialize the GaussianImageSampler with perturbations loaded from a file.

        Args:
        file_path (str): Path to the .pt file containing the perturbations.
        """
        self.data = data
        self.perturbations = self.load_perturbations()

        self.covariance_matrix = self.compute_covariance_matrix(sigma)
        self.sigma = self.compute_sigma(sigma)
        self.L = self.compute_cholesky()

    def load_perturbations(self):
        """
        Load perturbations from the .pt file.

        Returns:
        torch.Tensor: The perturbations tensor.
        """
        perturbations = self.data['perturbations']
        return perturbations
    
    def compute_sigma(self, sigma):
        """
        Compute Sigma.

        Returns:
        float: The perturbations tensor.
        """
        determinant = torch.det(self.covariance_matrix).item() + 1e-6 # numerical stability issues
        sigma = sigma / ( (sigma/determinant) ** (1/self.covariance_matrix.shape[0]) )
        return sigma


    def compute_covariance_matrix(self, sigma):
        """
        Compute the covariance matrix of the flattened perturbations.

        Returns:
        torch.Tensor: The covariance matrix.
        """
        # Compute the empirical mean and covariance matrix from the flattened tensor
        perturbations_flat = self.perturbations.view(self.perturbations.size(0), -1)

        # Compute the empirical covariance matrix using PyTorch
        empirical_cov = torch.cov(perturbations_flat.T)  # Transpose to have features as rows for covariance

        # Handle potential numerical instability with PyTorch
        eigenvalues = torch.linalg.eigvals(empirical_cov)
        min_eigenvalue = torch.min(eigenvalues.real)
        empirical_cov += torch.eye(empirical_cov.shape[0], device=empirical_cov.device) * (1e-6 - min_eigenvalue)

        # adjust to give the correct certifiaction guarantees
        determinant = torch.det(empirical_cov).item() + 1e-6 # numerical stability issues
        sigma = (sigma/determinant) ** (1/empirical_cov.shape[0])
        empirical_cov = empirical_cov * sigma

        return empirical_cov
    
    def compute_cholesky(self):
        """
        Compute the matrix L of the Cholesky decomposition of the covariance matrix.

        Returns:
        torch.Tensor: The matrix L of the Cholesky decomposition.
        """
        return torch.linalg.cholesky(self.covariance_matrix)
    
    def sample_multivariate_normal_gpu(self,n_samples):
        """
        Samples from a multivariate normal distribution using Cholesky decomposition on the GPU.

        Parameters:
        mean (torch.Tensor): Mean vector of the distribution, shape (d,).
        cov (torch.Tensor): Covariance matrix of the distribution, shape (d, d).
        n_samples (int): Number of samples to generate.

        Returns:
        samples (torch.Tensor): Samples from the multivariate normal distribution, shape (n_samples, d).
        """

        # Generate mean on the GPU   
        mean = torch.zeros(self.covariance_matrix.shape[0]).to(DEVICE)

        # Perform Cholesky decomposition of the covariance matrix
        L = self.L.to(DEVICE)

        # Generate standard normal samples on the GPU
        Z = torch.randn(n_samples, mean.shape[0], device=DEVICE)

        # Transform the standard normal samples
        samples = mean + Z @ L.T

        return samples

    def sample_smoothed_images(self, x, num_samples):
        """
        Sample new images from the Gaussian distribution.

        Args:
        x (torc.Tensor): A tensor that specifies the shape of the output
        num_samples (int): Number of images to sample.

        Returns:
        torch.Tensor: The sampled images, reshaped to original image dimensions.
        """

        # Generate batch of an original image
        batch = x.repeat((num_samples, 1, 1, 1))

        # Generate adversarial perturbations
        samples = self.sample_multivariate_normal_gpu(num_samples)#self.distribution.sample((num_samples,))
        sampled_images = samples.view(num_samples, *self.perturbations.shape[1:])

        # Generated smoothed images
        smoothed_images = batch.to(DEVICE) + sampled_images.to(DEVICE) * self.sigma
        smoothed_images = smoothed_images.clamp(0, 1)

        return smoothed_images
  
    def sample_perturbations(self, num_samples):
        """
        Sample new images from the Gaussian distribution.

        Args:
        x (torc.Tensor): A tensor that specifies the shape of the output
        num_samples (int): Number of images to sample.

        Returns:
        torch.Tensor: The sampled images, reshaped to original image dimensions.
        """

        # Generate batch of an original image
        perturbations = self.distribution.sample((num_samples,))
        perturbations = perturbations.view(num_samples, *self.perturbations.shape[1:])
        perturbations = perturbations.to(DEVICE)

        return perturbations
      


class GaussianSampler:
    def __init__(self, sigma):
        """
        Initialize the GaussianImageSampler.
        """

        self.sigma = sigma

    def sample_smoothed_images(self, x, num_samples):
        """
        Sample new images from the Gaussian distribution.

        Args:
        x (torc.Tensor): A tensor that specifies the shape of the output
        num_samples (int): Number of images to sample.

        Returns:
        torch.Tensor: The sampled images, reshaped to original image dimensions.
        """

        # Generate batch of an original image
        batch = x.repeat((num_samples, 1, 1, 1)).to(DEVICE)
        noise = torch.randn_like(batch, device=DEVICE) * self.sigma 
        smoothed_images = batch + noise
        smoothed_images = smoothed_images.clamp(0, 1)

        return smoothed_images