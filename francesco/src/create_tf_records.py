from preprocessing import data_loader

#data_loader.create_TF_records("dataset/original/", normalize=True)
data_loader.create_TF_records_with_half_manual("dataset/original/", normalize=True)
