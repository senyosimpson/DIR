import torch

class Reshape:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, sample):
        image1, image2 = sample
        image1 = image1.reshape(self.shape)
        image2 = image2.reshape(self.shape)
        return (image1, image2)


class Transpose:
    def __call__(self, sample):
        image1, image2 = sample
        image1 = image1.transpose((2,0,1))
        image2 = image2.transpose((2,0,1))
        return (image1, image2)


class ToTensor:
    def __call__(self, sample):
        '''
        Args:
            sample (tuple): A sample from the dataset
        
        Returns:
            sample (tuple(torch.Tensor, torch.Tensor)): Sample converted into a tensor
        '''
        image1, image2 = sample
        return (torch.Tensor(image1), torch.Tensor(image2))
    

class Normalize:
    def __call__(self, sample):
        '''
        Args:
            sample (tuple): A sample from the dataset
        Returns:
            sample (tuple): A sample with normalized image and corresponding label
        '''
        image1, image2 = sample
        image1 = 2 * (image1 - image1.min()) / (image1.max() - image1.min())-1
        image2 = 2 * (image2 - image2.min()) / (image2.max() - image2.min())-1
        return (image1, image2)