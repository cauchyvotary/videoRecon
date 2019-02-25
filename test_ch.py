
import chumpy as ch
import numpy as np
import cPickle as pkl
import scipy.sparse as sp
from chumpy.ch import Ch


class Func(Ch):
    dterms = 'x','y'

    def __init__(self, *args, **kwargs):
        self.on_changed(self._dirty_vars)

    def on_changed(self, which):
        if not hasattr(self, 'x'):
            self.x = ch.zeros(6)

        if not hasattr(self, 'y'):
            self.y = ch.zeros(6)

        self._set_up()

    def _set_up(self):
        self.v = self.x + self.y

    def compute_r(self):
        return self.v.r

    def compute_dr_wrt(self,wrt):
        if wrt is not self.x and wrt is not self.y:
            return  None
        return self.v.dr_wrt(wrt)

E={}

func = Func()
func.x.label = 'x'
func.y.label = 'y'
func.v.label = 'v'
func.label = 'func'
z = ch.zeros(3)
z.lable = 'z'
o = ch.zeros(3)
o.lable = 'o'
func.y[:] = np.ones(6)
func.x = ch.concatenate((z, o))
for i in range(2):


    E['func_{}'.format(i)] = (func -i)**2


ch.minimize(
    E,
    [z, o],
    method='dogleg'
)
print(func.x)
print(z)


