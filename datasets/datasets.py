from .data_layer import DataLayer
from .data_layer_ugen import DataLayer_uncertainty

def build_dataset(args, phase):
    data_layer = DataLayer
    return data_layer(args, phase)


def build_dataset_uncertainty(args, phase):
    data_layer = DataLayer_uncertainty
    return data_layer(args, phase)
