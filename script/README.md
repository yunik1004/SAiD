# Script

## Data Preprocessing

### [preprocess_voca_arkit.py](preprocess_voca_arkit.py)

It preprocesses the VOCA dataset to generate ARKit blendshapes.

```sh
optional arguments:
  --templates_dir TEMPLATES_DIR
                        Directory of the template meshes
  --blendshape_deltas_path BLENDSHAPE_DELTAS_PATH
                        Path of the blendshape deltas
  --blendshapes_out_dir BLENDSHAPES_OUT_DIR
                        Directory of output blendshapes
```

### [optimize_blendshape_coeffs.py](optimize_blendshape_coeffs.py)

It generates the pseudo-GT blendshape coefficients by solving the optimization problem.

```sh
optional arguments:
  --neutrals_dir NEUTRALS_DIR
                        Directory of the neutral meshes
  --blendshapes_dir BLENDSHAPES_DIR
                        Directory of the blendshape meshes
  --mesh_seqs_dir MESH_SEQS_DIR
                        Directory of the mesh sequences
  --blendshape_list_path BLENDSHAPE_LIST_PATH
                        List of the blendshapes
  --head_idx_path HEAD_IDX_PATH
                        List of the head indices
  --blendshapes_coeffs_out_dir BLENDSHAPES_COEFFS_OUT_DIR
                        Directory of the output coefficients
```

## Model Training

### [train.py](train.py)

It trains the SAiD model using VOCA-ARKit dataset.

```sh
optional arguments:
  --audio_dir AUDIO_DIR
                        Directory of the audio data
  --coeffs_dir COEFFS_DIR
                        Directory of the blendshape coefficients data
  --blendshape_deltas_path BLENDSHAPE_DELTAS_PATH
                        Path of the blendshape deltas
  --output_dir OUTPUT_DIR
                        Directory of the outputs
  --prediction_type PREDICTION_TYPE
                        Prediction type of the scheduler function, 'epsilon', 'sample', or 'v_prediction'
  --window_size WINDOW_SIZE
                        Window size of the blendshape coefficients sequence at training
  --batch_size BATCH_SIZE
                        Batch size at training
  --epochs EPOCHS       The number of epochs
  --learning_rate LEARNING_RATE
                        Learning rate
  --uncond_prob UNCOND_PROB
                        Unconditional probability of waveform (for classifier-free guidance)
  --ema EMA             Use Exponential Moving Average of models weights
  --val_period VAL_PERIOD
                        Period of validating model
  --val_repeat VAL_REPEAT
                        Number of repetition of val dataset
  --save_period SAVE_PERIOD
                        Period of saving model
```

## Evaluation

### [test_inference.py](test_inference.py)

It generates the inference outputs using VOCA-ARKit test dataset.

```sh
optional arguments:
  --weights_path WEIGHTS_PATH
                        Path of the weights of SAiD model
  --audio_dir AUDIO_DIR
                        Directory of the audio data
  --output_dir OUTPUT_DIR
                        Directory of the outputs
  --prediction_type PREDICTION_TYPE
                        Prediction type of the scheduler function, 'epsilon', 'sample', or 'v_prediction'
  --num_steps NUM_STEPS
                        Number of inference steps
  --strength STRENGTH   How much to paint
  --guidance_scale GUIDANCE_SCALE
                        Guidance scale
  --eta ETA             Eta for DDIMScheduler, between [0, 1]
  --fps FPS             FPS of the blendshape coefficients sequence
  --divisor_unet DIVISOR_UNET
                        Length of the blendshape coefficients sequence should be divided by this number
  --device DEVICE       GPU/CPU device
  --num_repeats NUM_REPEATS
                        Number of repetitions in inference for each audio
  --seed SEED           Random seed. Set the negative value if you don't want to control the randomness
```

### [test_evaluate.py](test_evaluate.py)

It evaluates the output based on the VOCA-ARKit test dataset

```sh
optional arguments:
  --audio_dir AUDIO_DIR
                        Directory of the audio data
  --coeffs_dir COEFFS_DIR
                        Directory of the blendshape coefficients data
  --blendshape_deltas_path BLENDSHAPE_DELTAS_PATH
                        Path of the blendshape deltas
  --sampling_rate SAMPLING_RATE
                        Sampling rate of the audio
  --fps FPS             FPS of the blendshape coefficients sequence
  --bc_threshold BC_THRESHOLD
                        Threshold for computing beat consistency score
  --output_dir OUTPUT_DIR
                        Directory of the outputs
```

## Inference

### [inference.py](inference.py)

It generates the inference result using the SAiD model

```sh
optional arguments:
  --weights_path WEIGHTS_PATH
                        Path of the weights of SAiD model
  --audio_path AUDIO_PATH
                        Path of the audio file
  --output_path OUTPUT_PATH
                        Path of the output blendshape coefficients file (csv format)
  --output_image_path OUTPUT_IMAGE_PATH
                        Path of the image of the output blendshape coefficients
  --intermediate_dir INTERMEDIATE_DIR
                        Saving directory of the intermediate outputs
  --prediction_type PREDICTION_TYPE
                        Prediction type of the scheduler function, 'epsilon', 'sample', or 'v_prediction'
  --save_intermediate SAVE_INTERMEDIATE
                        Save the intermediate outputs
  --num_steps NUM_STEPS
                        Number of inference steps
  --strength STRENGTH   How much to paint
  --guidance_scale GUIDANCE_SCALE
                        Guidance scale
  --eta ETA             Eta for DDIMScheduler, between [0, 1]
  --fps FPS             FPS of the blendshape coefficients sequence
  --divisor_unet DIVISOR_UNET
                        Length of the blendshape coefficients sequence should be divided by this number
  --device DEVICE       GPU/CPU device
  --init_sample_path INIT_SAMPLE_PATH
                        Path of the initial sample file (csv format)
  --mask_path MASK_PATH
                        Path of the mask file (csv format)
```