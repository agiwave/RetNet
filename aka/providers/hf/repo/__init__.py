import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import aka.numpy

def exists(*args, **kwargs):
    return join(*args, _raise_exceptions_for_missing_entries=False, **kwargs) is not None

def join(*args, _raise_exceptions_for_missing_entries=True, **kwargs):
    import transformers
    return transformers.utils.cached_file(*args, _raise_exceptions_for_missing_entries=_raise_exceptions_for_missing_entries, **kwargs)

def fileopen(repo, filename, **kwargs):
    return open(join(repo, filename, **kwargs))

def jsonload(repo, filename, **kwargs):
    import json
    return json.load(fileopen(repo, filename, **kwargs))

def safeopen(repo, filename, *, framework=aka.numpy.framework(), **kwargs):
    import safetensors
    return safetensors.safe_open(join(repo, filename, **kwargs), framework=framework)

def AutoDataset(*args, **kwargs):
    import datasets
    return datasets.load_dataset(*args, **kwargs)

def AutoModel(*args, **kwargs):
    import transformers
    return transformers.AutoModel.from_pretrained(*args, **kwargs)

def AutoTokenizer(*args, **kwargs):
    import transformers
    return transformers.AutoTokenizer.from_pretrained(*args, **kwargs)    

def AutoConfig(*args, **kwargs):
    import transformers
    return transformers.AutoConfig.from_pretrained(*args, **kwargs)  
