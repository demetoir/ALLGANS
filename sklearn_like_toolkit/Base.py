class BaseClass:
    def __str__(self):
        return self.__class__.__name__

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass
