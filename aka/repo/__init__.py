from .. import boot

def exists(repo, filename, **kwargs): return boot.invoke()
def join(repo, filename, **kwargs): return boot.invoke()

def fileopen(repo, filename, **kwargs): return boot.invoke()
def jsonload(repo, filename, **kwargs): return boot.invoke()
def safeopen(repo, filename, *, framework, **kwargs): return boot.invoke()

def AutoModel(repo): return boot.invoke()
def AutoDataset(repo): return boot.invoke()
def AutoConfig(repo): return boot.invoke()
def AutoTokenizer(repo): return boot.invoke()

boot.inject()