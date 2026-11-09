from settings import OUTPUT_DIR,FORMATS
from data_io import DataStore
from plot_style import Export

def context(ds=None,ex=None):
    if ds is None:ds=DataStore()
    if ex is None:ex=Export(OUTPUT_DIR,ds,FORMATS)
    return ds,ex
