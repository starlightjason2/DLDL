"""
Data preprocessing script for DLDL project.

WARNING: Memory-intensive. For large datasets, use cpu_use=0.2-0.3 and run
operations one at a time.
"""

import os
from util.preprocessor import Preprocessor
from constants import DATASET_DIR, SCALED_LABELS_FILENAME

preprocessor = Preprocessor(
    dataset_id="_meanvar-whole",
    cpu_use=0.2,
    normalization="meanvar-whole",
    labels_path=os.path.join(DATASET_DIR, SCALED_LABELS_FILENAME),
)
# Dataset is automatically created in __init__ if it doesn't exist
# Additional operations can be performed if needed
preprocessor.check_dataset(scale_labels=True)
# preprocessor_scaled.check_dataset(scale_labels=False, verbose=True)

# Convert labels to float32 (currently active operation)
preprocessor.convert_to_float()
