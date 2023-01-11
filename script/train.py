"""Train the SAiD_Wav2Vec2 model
"""
import numpy as np
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model
from said.model.diffusion import SAID_Wav2Vec2
from dataset import VOCARKitTrainDataset

if __name__ == "__main__":
    # Arguments
    batch_size = 2
    device = "cuda:0"

    # Load model with pretrained audio encoder
    said_model = SAID_Wav2Vec2()
    said_model.audio_encoder = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    said_model.to(device)

    audio_dir = "../data_train/audio"
    blendshape_coeffs_dir = "../data_train/blendshape_coeffs"
    train_dataset = VOCARKitTrainDataset(
        audio_dir, blendshape_coeffs_dir, said_model.sampling_rate, 120
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=VOCARKitTrainDataset.collate_fn,
    )

    data = next(iter(train_dataloader))
    waveform = data["waveform"]
    blendshape_coeffs = data["blendshape_coeffs"].to(device)

    waveform_processed = said_model.process_audio(waveform).to(device)
    random_timesteps = said_model.get_random_timesteps(batch_size).to(device)

    audio_embedding = said_model.get_audio_embedding(waveform_processed)
    noisy_coeffs = said_model.add_noise(blendshape_coeffs, random_timesteps)

    noise_pred = said_model(noisy_coeffs, random_timesteps, audio_embedding)

    print(f"Audio embedding: {audio_embedding.shape}")
    print(f"Noise_pred: {noise_pred.shape}")
