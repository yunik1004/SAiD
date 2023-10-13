# Script

## Data Preprocessing

- `preprocess_blendvoca.py`: Preprocess the VOCA dataset.
- `optimize_blendshape_coeffs.py`: Generate the pseudo-GT blendshape coefficients by solving the optimization problem.

## VAE Training/Inference

- `train_vae.py`: Train the VAE model using BlendVOCA dataset.
- `inference_vae.py`: Generate the inference result using VAE.

## SAiD Training/Inference

- `train.py`: Train the SAiD model using BlendVOCA dataset.
- `inference.py`: Generate the inference result using the SAiD model.

## Evaluation

- `test_inference.py`: Generate the inference outputs using BlendVOCA test dataset.
- `test_evaluate.py`: Evaluate the output based on the BlendVOCA test dataset.

## Rendering

- `render.py`: Render the blendshape coefficients into the video.
- `test_render.py`, `test_render.sh`: Render the BlendVOCA test data into the video.
