import flax.linen as ln
from jaxrk.core.typing import ConstOrInitFn
from jaxrk.core.init_fn import ConstFn
from typing import Callable

class Module(ln.Module):
    def const_or_param(self, name:str, init:ConstOrInitFn, *args, **kwargs):
        if isinstance(init, Callable):
            #initialization function
            return self.param(name, init, *args, **kwargs)
        else:
            #fixed float or array
            v = self.variable('constants', name, ConstFn(init, *args, **kwargs)) 
            return v.value