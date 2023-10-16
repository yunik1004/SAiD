# Data

## ARKit Reference

- `ARKit_blendshapes.txt`: 32 ARKit blendshapes' names.
- `ARKit_landmarks.txt`: Indices of the facial landmark vertices from the ARKit reference meshes.
- `ARKit_reference_blendshapes.zip`: ARKit reference meshes.

## VOCASET (FLAME)

- `FLAME_head_idx.txt`: Indices of the head vertices from the FLAME-based mesh.
- `FLAME_head_landmarks.txt`: Indices of the facial landmark vertices from the FLAME-based head mesh.
- `FLAME_landmarks.txt`: Indices of the facial landmark vertices from the FLAME-based mesh.

## BlendVOCA

- `coeffs_std.csv`: Standard deviations of blendshape coefficients over time.
- `blendshape_coeffs.zip`: Blendshape coefficients.
- `blendshape_residuals.pickle`: Blendshape residuals, with following structure:

    ```text
    {
        'FaceTalk_170725_00137_TA': {
            'jawForward': <np.ndarray object with shape (V, 3)>,
            ...
        },
        ...
    }
    ```
