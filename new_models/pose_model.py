
import chumpy as ch
import numpy as np
from chumpy.ch import Ch
from models.smpl import Smpl


class Model(Ch):
    terms = 'model',
    dterms = 'v_pose'

    def __init__(self, *args, **kwargs):
        self.on_changed(self._dirty_vars)

    def on_changed(self, which):

        if 'model' in which:
            dd = self.model
            self.v_origin = ch.array(dd['v_origin'])
            self.f = dd['f']
        if not hasattr(self, 'v_pose'):
            self.v_pose = ch.zeros_like(self.v_origin)
        self.v = self.v_origin + self.v_pose

    def compute_r(self):
        return self.v.r

    def compute_dr_wrt(self, wrt):
        if wrt is not self.v_pose:
            return None

        return self.v.dr_wrt(wrt)



if __name__ == '__main__':
    smpl = Smpl(model= '/home/suoxin/Body/videoavatars/vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    smpl.pose[:] = np.zeros(72)*0.1
    smpl.pose[0] = np.pi
    dd = {}

    dd['v_origin'] = smpl.v_template
    dd['f'] = smpl.f

    model = Model(dd)


    #model.v_pose = ch.random.rand(3)

    r0 = model.r

    E = (model - r0)
    ch.minimize(
        E,
        [model.v_pose],
        method='dogleg',
        options={'maxiter': 15, 'e_3': 0.001}
        # callback=get_cb(frames[0], base_smpl, camera, frustum) if display else None
    )


