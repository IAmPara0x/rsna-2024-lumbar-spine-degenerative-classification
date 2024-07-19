

### patient info structure

```json

{
    "patient_id": 0,

    "spinal_canal_stenosis_l1_l2": "Normal/Mild",
    "spinal_canal_stenosis_l2_l3": "Normal/Mild",
    "spinal_canal_stenosis_l3_l4": "Normal/Mild",
    "spinal_canal_stenosis_l4_l5": "Moderate",
    "spinal_canal_stenosis_l5_s1": "Normal/Mild",
    "left_neural_foraminal_narrowing_l1_l2": "Normal/Mild",
    "left_neural_foraminal_narrowing_l2_l3": "Normal/Mild",
    "left_neural_foraminal_narrowing_l3_l4": "Moderate",
    "left_neural_foraminal_narrowing_l4_l5": "Moderate",
    "left_neural_foraminal_narrowing_l5_s1": "Normal/Mild",
    "right_neural_foraminal_narrowing_l1_l2": "Normal/Mild",
    "right_neural_foraminal_narrowing_l2_l3": "Normal/Mild",
    "right_neural_foraminal_narrowing_l3_l4": "Normal/Mild",
    "right_neural_foraminal_narrowing_l4_l5": "Normal/Mild",
    "right_neural_foraminal_narrowing_l5_s1": "Normal/Mild",
    "left_subarticular_stenosis_l1_l2": "Normal/Mild",
    "left_subarticular_stenosis_l2_l3": "Moderate",
    "left_subarticular_stenosis_l3_l4": "Moderate",
    "left_subarticular_stenosis_l4_l5": "Severe",
    "left_subarticular_stenosis_l5_s1": "Normal/Mild",
    "right_subarticular_stenosis_l1_l2": "Normal/Mild",
    "right_subarticular_stenosis_l2_l3": "Normal/Mild",
    "right_subarticular_stenosis_l3_l4": "Normal/Mild",
    "right_subarticular_stenosis_l4_l5": "Severe",
    "right_subarticular_stenosis_l5_s1": "Normal/Mild",

    "series": {
        "<series_id>": {"series_description": "", "images": [ {"SOPInstanceUID": 1, "dicom": "<img>", "labels": ["<ImgLabel>"]} ]}
    }
}

```


## Explore

- "Image Position (Patient)" provides information about the scan Position.
- (pretrain?) Train the network to predict bounding box.
- Image captioning architecture(s) might be helpful when trying adding MRI scan positions information to the image.


## Benchmarks

- Baseline: 0.70913165807724

# TODO

- [X] prepare dataset
- [X] Train baseline model
