import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader, random_split
# Assuming BiGRUModel and EmbeddingDataset are defined as in previous examples
from Embedder import VideoEmbedder, LoadDF, load_dataset
import tqdm
import copy
import matplotlib.pyplot as plt
import json
from itertools import product
from argparse import ArgumentParser
# MLP Model Class for Scene Prediction
class ScenePredictionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ScenePredictionMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Modified Model Trainer Class for Integrated Training
class IntegratedModelTrainer:
    def __init__(self, audio_model, video_model, mlp_model, data_loader, learning_rate=0.001, num_epochs=5):
        self.audio_model = audio_model
        self.video_model = video_model
        self.mlp_model = mlp_model
        self.data_loader = data_loader
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self._init_optimizer()

    def _init_optimizer(self):
        # Include all parameters initially
        self.optimizer = torch.optim.Adam(
            list(self.audio_model.parameters()) + 
            list(self.video_model.parameters()) + 
            list(self.mlp_model.parameters()), 
            lr=self.learning_rate
        )

    def toggle_biGRU_trainability(self, trainable):
        # Freeze or unfreeze BiGRU models
        for param in self.audio_model.parameters():
            param.requires_grad = trainable
        for param in self.video_model.parameters():
            param.requires_grad = trainable

        # Reinitialize the optimizer with only MLP parameters if freezing BiGRU
        if not trainable:
            self.optimizer = torch.optim.Adam(self.mlp_model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            for audio_embeddings, video_embeddings, labels in self.data_loader:
                audio_output = self.audio_model(audio_embeddings)
                video_output = self.video_model(video_embeddings)

                combined_output = torch.cat((audio_output, video_output), dim=1)
                scene_predictions = self.mlp_model(combined_output)

                loss = self.criterion(scene_predictions, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')