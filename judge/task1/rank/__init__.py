import os

os.environ.setdefault("_RJEM_MALLOC_CONF", "background_thread:true,dirty_decay_ms:500,muzzy_decay_ms:500")
os.environ.setdefault("POLARS_MAX_THREADS", "4")
