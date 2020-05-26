def error_handler(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BaseException as error:
            print(error)
    return wrapper
