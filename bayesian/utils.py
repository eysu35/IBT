import functools


def method_cache(fn):
    cache_attr = "_cache_" + fn.__name__
    cache_info_attr = "_cache_info_" + fn.__name__

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
            setattr(self, cache_info_attr, dict(hits=0, misses=0))
        cache = getattr(self, cache_attr)
        cache_info = getattr(self, cache_info_attr)

        key = (args, frozenset(kwargs.items()) if kwargs else None)
        if key not in cache:
            cache[key] = fn(self, *args, **kwargs)
            cache_info["misses"] += 1
        cache_info["hits"] += 1
        return cache[key]

    return wrapped
