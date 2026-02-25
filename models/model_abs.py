from abc import ABC,abstractmethod

class Model(ABC):
    @abstractmethod
    def __init__(self,**kwargs):
        pass
    @abstractmethod
    def infer(self,pkl_path_dict_list:list):
        pass
