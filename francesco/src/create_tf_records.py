from preprocessing import data_loader

#data_loader.create_TF_records("dataset/original/", normalize=True)
#data_loader.create_TF_records("dataset/original/", normalize=False)
data_loader.create_TF_records("dataset/augmented/", normalize=False)
