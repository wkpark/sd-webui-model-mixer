import torch
import numpy as np


def image_embeddings_direct(image, model, processor, use_cuda=True):
    inputs = processor(images=image, return_tensors='pt')['pixel_values']
    if use_cuda:
        inputs = inputs.to('cuda')
        model.to("cuda")
    result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()

    if use_cuda:
        model.to("cpu")
    return (result / np.linalg.norm(result)).squeeze(axis=0)


class Classifier(torch.nn.Module):
    """binary classifier that consumes CLIP embeddings"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = torch.nn.Linear(hidden_size//2, output_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x
