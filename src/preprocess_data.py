from util.preprocessor import Preprocessor
from constants import DATA_DIR, DATASET_DIR, LABELS_PATH

preprocessor = Preprocessor(
    DATASET_DIR, DATA_DIR, LABELS_PATH, dataset_id="_meanvar-whole"
)
# preprocessor_scaled = Preprocessor(DATASET_DIR, DATA_DIR, LABELS_PATH, dataset_id='_scaled_labels')
# labels = preprocessor.make_labels_naive(save=True)
# labels = preprocessor_scaled.make_labels_scaled(save=True)
# stats = preprocessor.get_mean_std(cpu_use=1)
# labels = preprocessor.make_labels_scaled(save=True)
# preprocessor.make_dataset(normalization='meanvar-whole', make_labels=False, cpu_use=1)
# preprocessor.check_dataset(labels_path=DATASET_DIR+'processed_labels_scaled_labels.pt',\
#        normalization='meanvar-whole', scale_labels=True)
# preprocessor_scaled.check_dataset(scale_labels=False, verbose=True)
preprocessor.convert_to_float(
    labels_path=DATASET_DIR + "processed_labels_scaled_labels.pt"
)
