from utils.logging.tf_scalar_logger import TFScalarLogger


class TFScalarLoggerFactory:

    def __init__(self, name, minimal=False):
        self.name = name
        self.minimal = minimal


    def __call__(self, *args, **kwds):
        return TFScalarLogger(name=self.name, minimal=self.minimal, *args, **kwds)
