from preprocessor import Preprocessor

data_dir = '/eagle/fusiondl_aesp/signal_data/d3d/ipspr15V/'
dataset_dir = '/eagle/fusiondl_aesp/jrodriguez/processed_data/'
labels_path = '/eagle/fusiondl_aesp/jrodriguez/shot_lists/ips_labels.txt'

preprocessor = Preprocessor(dataset_dir, data_dir, labels_path, dataset_id='_meanvar-whole')
#preprocessor_scaled = Preprocessor(dataset_dir, data_dir, labels_path, dataset_id='_scaled_labels')
#labels = preprocessor.make_labels_naive(save=True)
#labels = preprocessor_scaled.make_labels_scaled(save=True)
#stats = preprocessor.get_mean_std(cpu_use=1)
#labels = preprocessor.make_labels_scaled(save=True)
#preprocessor.make_dataset(normalization='meanvar-whole', make_labels=False, cpu_use=1)
#preprocessor.check_dataset(labels_path=dataset_dir+'processed_labels_scaled_labels.pt',\
#        normalization='meanvar-whole', scale_labels=True)
#preprocessor_scaled.check_dataset(scale_labels=False, verbose=True)
preprocessor.convert_to_float(labels_path=dataset_dir+'processed_labels_scaled_labels.pt')
