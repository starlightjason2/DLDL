"""
Data preprocessing script for DLDL project.

This script provides examples of how to use the Preprocessor class to:
- Compute dataset statistics (max length, mean/std)
- Create labels (naive or scaled)
- Build preprocessed datasets with various normalization strategies
- Validate preprocessed datasets
- Convert tensors to float32

Uncomment the operations you want to perform. Most preprocessing steps
are one-time operations that generate files for later use.
"""

from util.preprocessor import Preprocessor
from constants import DATA_DIR, DATASET_DIR, LABELS_PATH

# Initialize preprocessor for mean-variance normalized dataset
preprocessor = Preprocessor(
    DATASET_DIR, DATA_DIR, LABELS_PATH, dataset_id="_meanvar-whole"
)

# Alternative preprocessor configuration (commented out)
# preprocessor_scaled = Preprocessor(DATASET_DIR, DATA_DIR, LABELS_PATH, dataset_id='_scaled_labels')

# Example preprocessing operations (uncomment as needed):
# labels = preprocessor.make_labels_naive(save=True)
# labels = preprocessor_scaled.make_labels_scaled(save=True)
# stats = preprocessor.get_mean_std(cpu_use=1)
# labels = preprocessor.make_labels_scaled(save=True)
# preprocessor.make_dataset(normalization='meanvar-whole', make_labels=False, cpu_use=1)
# preprocessor.check_dataset(labels_path=DATASET_DIR+'processed_labels_scaled_labels.pt',\
#        normalization='meanvar-whole', scale_labels=True)
# preprocessor_scaled.check_dataset(scale_labels=False, verbose=True)

# Convert labels to float32 (currently active operation)
preprocessor.convert_to_float(
    labels_path=DATASET_DIR + "processed_labels_scaled_labels.pt"
)
